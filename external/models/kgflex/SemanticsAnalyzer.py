import tqdm

from .UserFeatureMapper import UserFeatureMapper
from collections import defaultdict
import numpy as np
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
import pandas as pd
import random
import os


class SemanticsAnalyzer(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        # auto parameters
        self._params_list = [
            ("_parallel_ufm", "parallel_ufm", "pufm", 8, int, None),
            ("_first_order_limit", "first_order_limit", "fol", -1, None, None),
            ("_second_order_limit", "second_order_limit", "sol", -1, None, None),
            ("_predicate_mapping", "predicate_mapping", "pm", None, None, None),
            ("_object_mapping", "object_mapping", "om", None, None, None),
            ("_save_path", "save_path", "sp", None, None, None),
            ("_seed", "seed", "seed", 42, None, None),
            ("_npr", "npr", "npr", 1, None, None),
            ("_criterion", "criterion", "c", 'info_gain', str, None)
        ]
        self.autoset_params()
        np.random.seed(self._seed)
        random.seed(self._seed)

        self.name = f'SemanticsAnalyzer_{self.get_params_shortcut()}'

        training_set = self._data.train_pd
        self.transactions = len(training_set)

        # ------------------------------ ITEM FEATURES ------------------------------
        print('ITEM FEATURES: loading items features...\n')
        self.item_features_mapper = {item: set(map(tuple,
                                                   self._data.side_information_data.feature_map[
                                                       self._data.side_information_data.feature_map.itemId ==
                                                       self._data.private_items[item]]
                                                   [['predicate', 'object']].values))
                                     for item in self._data.private_items}
        print(f'\nITEM FEATURES: features loaded for {len(self.item_features_mapper)} items\n')

        # ------------------------------ USER FEATURES ------------------------------
        print('USER FEATURES: user features loading...\n')
        self.user_feature_mapper = UserFeatureMapper(self._data,
                                                     self.item_features_mapper,
                                                     self._data.side_information_data.predicate_mapping,
                                                     self._first_order_limit,
                                                     self._second_order_limit,
                                                     npr=self._npr,
                                                     random_seed=self._seed,
                                                     n_procs=self._parallel_ufm,
                                                     depth=2,
                                                     criterion=self._criterion)
        client_ids = list(self._data.i_train_dict.keys())
        self.user_feature_mapper.user_feature_weights(client_ids)

        feat = defaultdict(int)

        pred = pd.read_csv(self._predicate_mapping, names=['p', 'id', 'order'], sep="\t")
        obj = pd.read_csv(self._object_mapping, names=['o', 'id'], sep="\t")

        # for c, user_id in tqdm.tqdm(enumerate(client_ids), total=client_ids):
        #     for i, v in enumerate(self.user_feature_mapper.client_features[user_id]):
        #         feat['<' + ', '.join(el.split('/')[-1] for el in pred.loc[v[0]]['p'].split('~')) + ', ' +
        #            obj.loc[v[1]]['o'].split('/')[-1].split(':')[-1] + '>'] += self.user_feature_mapper.client_features[user_id][v]

        for c, user_id in tqdm.tqdm(enumerate(client_ids), total=client_ids):
            for i, v in enumerate(self.user_feature_mapper.client_features[user_id]):
                feat[v] += self.user_feature_mapper.client_features[user_id][v]

        k = pd.DataFrame.from_dict(feat, orient="index")
        k[0] = k[0].apply(lambda x: int(x * 100))
        k = k.reset_index()
        k['word'] = k['index']
        k['color'] = pd.Series()
        k['url'] = pd.Series()
        k = k.drop(columns=['index'])
        k.to_csv(os.path.join(self._save_path, 'features_word_cloud.csv'), index=False,
                 header=['weight', 'word', 'color', 'url'], sep=";")
        print("File saved. Ignore the following errors.")
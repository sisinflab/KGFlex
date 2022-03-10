import random

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import math

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin

from .UserFeatureMapper import UserFeatureMapper
from . import Client, ClientModel, Server, ServerModel
from collections import defaultdict


class KGFlex(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        # auto parameters
        self._params_list = [
            ("_q", "q", "q", 0.1, None, None),
            ("_lr", "lr", "lr", 0.01, None, None),
            ("_upc", "updates_per_client", "upc", None, None, None),
            ("_embedding", "embedding", "em", 10, int, None),
            ("_parallel_ufm", "parallel_ufm", "pufm", 0, int, None),
            ("_first_order_limit", "first_order_limit", "fol", -1, None, None),
            ("_second_order_limit", "second_order_limit", "sol", -1, None, None),
            ("_centralized", "centralized", "centralized", -1, None, None),
            ("_batch_size", "batch_size", "batch_size", 1024, int, None),
            ("_seed", "seed", "seed", 42, None, None),
            ("_npr", "npr", "npr", 1, None, None),
            ("_criterion", "criterion", "c", 'info_gain', str, None)
        ]
        self.autoset_params()
        np.random.seed(self._seed)
        random.seed(self._seed)

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
        print('USER FEATURES: user features loaded\n')

        # ------------------------------ MODEL FEATURES ------------------------------
        print('MODEL FEATURES: features mapping...')
        # mapping features in columns
        model_features_mapping = defaultdict(lambda: len(model_features_mapping))
        for c in client_ids:
            for feature in self.user_feature_mapper.client_features[c]:
                _ = model_features_mapping[feature]
        print(f'MODEL FEATURES: {len(model_features_mapping)} features mapped')

        # total number of features (i.e. columns of the item matrix / latent factors)
        item_features_mask = []
        for _, v in self.item_features_mapper.items():
            common = set.intersection(set(model_features_mapping.keys()), set(v))
            item_features_mask.append([True if f in common else False for f in model_features_mapping])
        self.item_features_mask = csr_matrix(item_features_mask)

        # ------------------------------ SYSTEM INITIALIZATION ------------------------------

        print('SYSTEM INFO: initializing server...')
        self.server = Server.Server(self._lr, ServerModel.ServerModel(model_features_mapping, self._embedding, self._seed))
        print('SYSTEM INFO: server initialized\n')


        print('SYSTEM INFO: initializing clients...\n')
        self.clients = [
            Client.Client(c, ClientModel.ClientModel(self._embedding),
                          self._data.i_train_dict[c], self.user_feature_mapper.client_features[c],
                          model_features_mapping, self._upc,
                          self.item_features_mask, self._seed) for c in tqdm(client_ids)]
        print(f"\nSYSTEM INFO: {len(self.clients)} clients initialized\n")

    @property
    def name(self):
        return "KGFlex" \
               + "_e:" + str(self._epochs) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in range(self._epochs):
            if self._centralized == 1:
                print("centralized training")
                self.server.centralized_training(self.transactions, self.clients, self._batch_size)
            else:
                selected_clients = list(
                    np.random.choice(self.clients, math.ceil(self._q * len(self.clients)), replace=False))
                self.server.train_model(selected_clients)
            self.evaluate(it)

    def get_recommendations(self, k: int = 100):
        if not self._negative_sampling:
            return {}, self.server.predict(self.clients, self._data.private_users, self._data.private_items,
                                           mask=self.get_candidate_mask(), max_k=k)
        else:
            return self.server.predict(self.clients, self._data.private_users, self._data.private_items,
                                       mask=self.get_candidate_mask(validation=True), max_k=k) if hasattr(self._data,
                                                                                                          "val_dict") else {}, \
                   self.server.predict(self.clients, self._data.private_users, self._data.private_items,
                                       mask=self.get_candidate_mask(), max_k=k)

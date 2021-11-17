import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import math

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

from .UserFeatureMapper import UserFeatureMapper
from . import Client, ClientModel, Server, ServerModel
from collections import defaultdict
from .kgflex_model import KGFlex_Model

class KGFlex(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        # auto parameters
        self._params_list = [
            ("_q", "q", "q", 0.1, None, None),
            ("_lr", "lr", "lr", 0.01, None, None),
            ("_upc", "updates_per_client", "upc", None, None, None),
            ("_embedding", "embedding", "em", 10, int, None),
            ("_parallel_ufm", "parallel_ufm", "pufm", 8, int, None),
            ("_first_order_limit", "first_order_limit", "fol", -1, None, None),
            ("_second_order_limit", "second_order_limit", "sol", -1, None, None),
            ("_centralized", "centralized", "centralized", -1, None, None),
            ("_batch_size", "batch_size", "batch_size", 1024, int, None),
            ("_seed", "seed", "seed", 42, None, None)
        ]
        self.autoset_params()
        np.random.seed(self._seed)

        training_set = self._data.train_pd
        self.transactions = len(training_set)

        # ------------------------------ ITEM FEATURES ------------------------------
        print('importing items features')

        self.item_features_mapper = {item: set(map(tuple,
                                                   self._data.side_information_data.feature_map[
                                                       self._data.side_information_data.feature_map.itemId ==
                                                       self._data.private_items[item]]
                                                   [['predicate', 'object']].values))
                                     for item in self._data.private_items}

        # ------------------------------ USER FEATURES ------------------------------
        print('user features loading')

        self.user_feature_mapper = UserFeatureMapper(self._data.i_train_dict,
                                                     self.item_features_mapper,
                                                     self._data.side_information_data.predicate_mapping, self._seed)
        client_ids = list(self._data.i_train_dict.keys())

        users = list(self._data.private_users.keys())
        items = list(self._data.private_items.keys())
        n_users = len(users)
        n_items = len(items)

        self.user_feature_mapper.compute_and_export_features(client_ids, self._parallel_ufm, self._first_order_limit,
                                                             self._second_order_limit)

        # ------------------------------ MODEL FEATURES ------------------------------
        print('features mapping')
        users_features = self.user_feature_mapper.client_features

        # mapping features in columns
        features_mapping = defaultdict(lambda: len(features_mapping))
        for c in client_ids:
            for feature in users_features[c]:
                _ = features_mapping[feature]

        # total number of features (i.e. columns of the item matrix / latent factors)
        print('FEATURES INFO: {} features found'.format(len(features_mapping)))
        item_features_mask = []
        for _, v in self.item_features_mapper.items():
            common = set.intersection(set(features_mapping.keys()), set(v))
            item_features_mask.append([True if f in common else False for f in features_mapping])
        self.item_features_mask = csr_matrix(item_features_mask)

        index_mask = {user: [True if f in users_features[user] else False
                             for f in features_mapping] for user in users}

        index_mask_csr = {user: csr_matrix(mask) for user, mask in index_mask.items()}

        user_item_feature_mask = {user: csr_matrix.multiply(self.item_features_mask, index_mask[0])
                                  for user, mask in index_mask.items()}

        # ------------------------------ POSITIVE AND NEGATIVE ITEMS ------------------------------

        # positive items are the one in the training set
        positive_items = {user: set(ratings.keys()) for user, ratings in self._data.i_train_dict.items()}
        all_items = set(self._data.private_items)
        negative_items = {user: all_items - items for user, items in positive_items.items()}


        # ------------------------------ MODEL ------------------------------

        model = KGFlex_Model(learning_rate=self._lr,
                             n_users=n_users,
                             users=users,
                             n_items=n_items,
                             n_features=len(features_mapping),
                             features_mapping=features_mapping,
                             embedding_size=self._embedding,
                             positive_items=positive_items,
                             negative_items=negative_items,
                             index_mask=index_mask,
                             users_features=users_features)

        # ------------------------------ INITIALIZING NODES ------------------------------
        print('initializing server')
        self.server = Server.Server(self._lr,
                                    ServerModel.ServerModel(features_mapping, self._embedding, self._seed))

        print('creating clients')
        self.clients = [
            Client.Client(c, ClientModel.ClientModel(self._embedding),
                          self._data.i_train_dict[c], self.user_feature_mapper.client_features[c],
                          features_mapping, self._upc,
                          self.item_features_mask, self._seed) for c in tqdm(client_ids)]

        print(f"\nINFO: clients created\n")

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

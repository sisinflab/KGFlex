import math
import random
import numpy as np
from itertools import islice
from operator import itemgetter
from collections import OrderedDict, Counter
import pandas as pd
from multiprocessing import cpu_count, Pool
import multiprocessing

from sys import platform
if platform == 'darwin':
    multiprocessing.set_start_method("fork")

import tqdm


class UserFeatureMapper:
    def __init__(self, data, item_features, predicate_mapping: pd.DataFrame,
                 n_first_order_features, n_second_order_features, npr=1, n_procs=0, random_seed=42, depth=2, criterion='info_gain'):

        # set random seeds
        np.random.seed(random_seed)
        random.seed(random_seed)

        self._data = data
        self._items = set(self._data.private_items.keys())

        # first order and second order predicates
        fo_predicates = set(predicate_mapping[predicate_mapping['predicate_order'] == 1]['predicate'].to_list())
        so_predicates = set(predicate_mapping[predicate_mapping['predicate_order'] == 2]['predicate'].to_list())
        self.predicates = [fo_predicates, so_predicates]

        # number of features for each exploration depth
        self.limits = [n_first_order_features, n_second_order_features]
        self.depth = depth

        assert depth == len(self.limits)
        assert depth == len(self.predicates)

        # number of parallel processes
        if n_procs == 0:
            n_procs = cpu_count() - 1
        self._n_procs = n_procs
        assert n_procs > 0
        print(f'USER FEATURE MAPPER: {n_procs} parallel processes')

        # declaring variables
        self.client_features = dict()
        self.item_features = item_features
        self.npr = npr
        print(f'USER FEATURE MAPPPER: negative-positive ratio set to {npr}')
        self.criterion = criterion

    def user_feature_weights(self, client_ids):

        if self._n_procs == 1:
            return self.get_user_feature_weights(client_ids)
        else:
            return self.get_user_feature_weights_mp(client_ids)

    def get_user_feature_weights_mp(self, client_ids):

        def args():
            return ((set(self._data.i_train_dict[c].keys()),
                     set.difference(self._items, self._data.i_train_dict[c].keys()),
                     self.npr,
                     self.item_features,
                     self.predicates,
                     self.limits,
                     self.criterion) for c in client_ids)

        arguments = args()
        pool = Pool(processes=self._n_procs)
        results = pool.starmap(user_feature_weights,
                               tqdm.tqdm(arguments, total=len(client_ids), desc='users features weights'))
        pool.close()
        pool.join()

        for c, i in enumerate(client_ids):
            self.client_features[c] = results[i]

    def get_user_feature_weights(self, client_ids):

        # compute for each client
        self.client_features = {c: user_feature_weights(
            positive_items=self._data.i_train_dict[c].keys(),
            total_negative_items=set.difference(self._items, self._data.i_train_dict[c].keys()),
            npr=self.npr,
            item_features=self.item_features,
            predicates=self.predicates,
            limits=self.limits,
            weight_type=self.criterion)
            for c in tqdm.tqdm(client_ids, desc='users features weights')}


def user_feature_weights(positive_items, total_negative_items, npr, item_features, predicates, limits, weight_type='info_gain'):
    # select client negative items
    negative_items = random_pick(total_negative_items, len(positive_items) * npr, strategy='popularity')
    # count features
    pos_counter, neg_counter = user_features_counter(positive_items, negative_items, item_features)
    # compute information metric over features
    features = compute_feature_weights(pos_counter, neg_counter, predicates, len(positive_items),
                                       len(positive_items) * npr, limits=limits, weight_type=weight_type)
    # return client features
    return features


def user_features_counter(positive_items, negative_items, item_features_):
    """
    Given a list of positive and negative items counts all them features
    :param positive_items: list positive items
    :param negative_items: list of negative items
    :param item_features_: dictionary containing all the item features
    :return:
    """

    def count_features(items):
        features = []
        for i in items:
            features.extend(item_features_.get(i, set()))
        return Counter(features)

    pos_counter = count_features(positive_items)
    neg_counter = count_features(negative_items)

    return pos_counter, neg_counter


def compute_feature_weights(positive_counter, negative_counter, predicates, n_positive_items, n_negative_items,
                            limits, depth=2, weight_type='info_gain'):
    assert len(limits) == depth

    # split features (in positive and negative items) respect to their depth
    pos = [Counter({f: c for f, c in positive_counter.items() if f[0] in predicates[d]}) for d in range(depth)]
    neg = [Counter({f: c for f, c in negative_counter.items() if f[0] in predicates[d]}) for d in range(depth)]

    # select best features for each exploration depth
    selected_features = OrderedDict()
    for d, limit in zip(range(depth), limits):
        if limit == -1 or limit >= len(pos[d]):
            # select all
            features = feature_weight(pos[d], neg[d], n_positive_items, n_negative_items, type_=weight_type)
        else:
            # select top-limit features
            features = feature_weight(pos[d], neg[d], n_positive_items, n_negative_items, type_=weight_type)
            features = OrderedDict(islice(features.items(), limit))
        selected_features.update(features)

    return selected_features


def feature_weight(positive_counter, negative_counter, n_positive_items, n_negative_items, type_='info_gain',
                   threshold=0):
    if type_ == 'gini':
        features = feature_gini(positive_counter, negative_counter, n_positive_items, n_negative_items, threshold)
    else:
        features = feature_info_gain(positive_counter, negative_counter, n_positive_items, n_negative_items, threshold)
    return OrderedDict(sorted(features.items(), key=itemgetter(1, 0), reverse=True))


def gini_impurity(positive, negative, total):
    """
    Gini impurity for two classes (feature in positive items and feature in negative items)
    """
    # source:
    #   "Ranking with decision tree", Fen Xia et al.
    #   DOI 10.1007/s10115-007-0118-y
    # Gini = p_positive ( 1 - p_positive ) + p_negative ( 1 - p_negative )
    # p_positive = positive / total
    # p_negative = negative / total
    # 1 - p_positive = p_negative | 1 - p_negative = p_positive
    if total == 0:
        return 0
    else:
        return 2 * positive * negative / (total ** 2)


def feature_gini(positive_counter, negative_counter, n_positive_items, n_negative_items=None, threshold=0):
    if n_negative_items is None:
        # if not specified, it is supposed that positive and negative items are balanced
        n_negative_items = n_positive_items

    # weighted version of the gini impurity
    ratio = n_positive_items / n_negative_items

    # total number of items
    # total = (n_positive_items / ratio) + n_negative_items
    # equivalent formulation of total
    total = n_negative_items * 2

    feature_gini = dict()
    # assumption: n positive items == n negative items -> gini = 2 * 1/2 * 1/2 = 0.5
    total_gini = 0.5

#    for feature in set.union(set(positive_counter), set(negative_counter)):
    for feature in set(positive_counter):
        pos = positive_counter[feature] / ratio
        neg = negative_counter[feature]
        p = (pos + neg) / total

        gini = total_gini - p * gini_impurity(pos, neg, pos + neg) - (1-p) * gini_impurity(n_negative_items - pos, n_negative_items - neg, total - (pos + neg))
        if gini > threshold:
            feature_gini[feature] = gini
    return feature_gini


def info_gain(positive, negative, n_positive_items, n_negative_items=None):
    def relative_gain(partial, total):
        if total == 0:
            return 0
        ratio = partial / total
        if ratio == 0:
            return 0
        return - ratio * math.log2(ratio)

    if n_negative_items is None:
        # if not specified, it is supposed that positive and negative items are balanced
        n_negative_items = n_positive_items

    den_1 = positive + negative

    h_present = relative_gain(positive, den_1) + relative_gain(negative, den_1)
    den_2 = n_positive_items + n_negative_items - (positive + negative)

    num_1 = n_positive_items - positive
    num_2 = n_negative_items - negative
    h_absent = relative_gain(num_1, den_2) + relative_gain(num_2, den_2)

    return 1 - den_1 / (den_1 + den_2) * h_present - den_2 / (den_1 + den_2) * h_absent


def feature_info_gain(positive_counter, negative_counter, n_positive_items, n_negative_items=None, threshold=0):
    if n_negative_items is None:
        # if not specified, it is supposed that positive and negative items are balanced
        n_negative_items = n_positive_items

    # weighted version of the information gain
    ratio = n_positive_items / n_negative_items

    # weighted value of n_positive_items = n_positive_items / ratio = n_negative_items

    feature_igs = dict()

    for feature in set(positive_counter):
        ig = info_gain(positive_counter[feature] / ratio, negative_counter[feature], n_negative_items,
                       n_negative_items)
        if ig > threshold:
            feature_igs[feature] = ig

    return feature_igs


def random_pick(source, n, strategy='uniform'):
    def pick_uniform():
        """
        Pick n elements.
        Every element is picked with the same probability
        :return: set of picked elements
        """
        return random.choices(list(set(source)), k=n)

    def pick_with_popularity():
        """
        Pick n elements.
        Popular elements have more chances to be picked.
        :return: set of picked elements
        """
        return random.choices(source, k=n)

    source = list(source)

    if strategy == 'uniform':
        return pick_uniform()
    else:
        return pick_with_popularity()

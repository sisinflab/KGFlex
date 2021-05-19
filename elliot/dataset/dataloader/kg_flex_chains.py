"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import os
import numpy as np
import typing as t
import pandas as pd
import scipy.sparse as sp
from collections import Counter
from enum import Enum
from types import SimpleNamespace
import logging as pylog

from elliot.negative_sampling.negative_sampling import NegativeSampler
from elliot.utils import logging
from elliot.splitter.base_splitter import Splitter
from elliot.prefiltering.standard_prefilters import PreFilter

"""
[(train_0,test_0)]
[([(train_0,val_0)],test_0)]
[data_0]

[([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_0),
([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_1),
([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_2),
([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_3),
([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_4)]

[[data_0,data_1,data_2,data_3,data_4],
[data_0,data_1,data_2,data_3,data_4],
[data_0,data_1,data_2,data_3,data_4],
[data_0,data_1,data_2,data_3,data_4],
[data_0,data_1,data_2,data_3,data_4]]

[[data_0],[data_1],[data_2]]

[[data_0,data_1,data_2]]

[[data_0,data_1,data_2],[data_0,data_1,data_2],[data_0,data_1,data_2]]
"""

class TextColor(Enum):
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class KGFlexLoader:
    """
    Load train and test dataset
    """

    def __init__(self, config, *args, **kwargs):
        """
        Constructor of DataSet
        :param path_train_data: relative path for train file
        :param path_test_data: relative path for test file
        """

        self.logger = logging.get_logger(self.__class__.__name__)
        self.args = args
        self.kwargs = kwargs
        self.config = config
        self.column_names = ['userId', 'itemId', 'rating', 'timestamp']
        if config.config_test:
            return

        self.side_information_data = SimpleNamespace()

        if config.data_config.strategy == "fixed":
            path_train_data = config.data_config.train_path
            path_val_data = getattr(config.data_config, "validation_path", None)
            path_test_data = config.data_config.test_path

            work_directory_path = config.data_config.side_information.work_directory
            map_path = config.data_config.side_information.map
            features_path = config.data_config.side_information.features
            predicates_path = config.data_config.side_information.predicates

            self.train_dataframe, self.side_information_data.feature_map, self.side_information_data.predicate_mapping = self.load_dataset_dataframe(path_train_data,
                                                                                                 predicates_path,
                                                                                                 features_path)

            self.train_dataframe = self.check_timestamp(self.train_dataframe)

            self.logger.info(f"{path_train_data} - Loaded")

            self.test_dataframe = pd.read_csv(path_test_data, sep="\t", header=None, names=self.column_names)

            self.test_dataframe = self.check_timestamp(self.test_dataframe)

            if config.binarize == True:
                self.test_dataframe["rating"] = 1
                self.train_dataframe["rating"] = 1

            if path_val_data:
                self.validation_dataframe = pd.read_csv(path_val_data, sep="\t", header=None, names=self.column_names)
                self.validation_dataframe = self.check_timestamp(self.validation_dataframe)

                self.tuple_list = [([(self.train_dataframe, self.validation_dataframe)], self.test_dataframe)]
            else:
                self.tuple_list = [(self.train_dataframe, self.test_dataframe)]

        elif config.data_config.strategy == "hierarchy":
            item_mapping_path = getattr(config.data_config.side_information, "item_mapping", None)
            self.side_information_data.feature_map = self.load_attribute_file(item_mapping_path)

            self.tuple_list = self.read_splitting(config.data_config.root_folder)

            self.logger.info('{0} - Loaded'.format(config.data_config.root_folder))

        elif config.data_config.strategy == "dataset":
            self.logger.info("There will be the splitting")
            path_dataset = config.data_config.dataset_path

            work_directory_path = config.data_config.side_information.work_directory
            map_path = config.data_config.side_information.map
            features_path = config.data_config.side_information.features
            predicates_path = config.data_config.side_information.predicates

            self.dataframe, self.side_information_data.feature_map, self.side_information_data.predicate_mapping = self.load_dataset_dataframe(path_dataset,
                                                                                                 predicates_path,
                                                                                                 features_path)
            self.dataframe = self.check_timestamp(self.dataframe)

            self.logger.info(('{0} - Loaded'.format(path_dataset)))

            self.dataframe = PreFilter.filter(self.dataframe, self.config)

            if config.binarize == True:
                self.dataframe["rating"] = 1

            splitter = Splitter(self.dataframe, self.config.splitting)
            self.tuple_list = splitter.process_splitting()

        else:
            raise Exception("Strategy option not recognized")

    def check_timestamp(self, d: pd.DataFrame) -> pd.DataFrame:
        if all(d["timestamp"].isna()):
            d = d.drop(columns=["timestamp"]).reset_index(drop=True)
        return d

    def read_splitting(self, folder_path):
        tuple_list = []
        for dirs in os.listdir(folder_path):
            for test_dir in dirs:
                test_ = pd.read_csv(f"{folder_path}{test_dir}/test.tsv", sep="\t")
                val_dirs = [f"{folder_path}{test_dir}/{val_dir}/" for val_dir in os.listdir(f"{folder_path}{test_dir}") if os.path.isdir(f"{folder_path}{test_dir}/{val_dir}")]
                val_list = []
                for val_dir in val_dirs:
                    train_ = pd.read_csv(f"{val_dir}/train.tsv", sep="\t")
                    val_ = pd.read_csv(f"{val_dir}/val.tsv", sep="\t")
                    val_list.append((train_, val_))
                if not val_list:
                    val_list = pd.read_csv(f"{folder_path}{test_dir}/train.tsv", sep="\t")
                tuple_list.append((val_list, test_))

        return tuple_list

    def generate_dataobjects(self) -> t.List[object]:
        data_list = []
        for train_val, test in self.tuple_list:
            # testset level
            if isinstance(train_val, list):
                # validation level
                val_list = []
                for train, val in train_val:
                    single_dataobject = KGFlexDataObject(self.config, (train,val,test), self.side_information_data, self.args, self.kwargs)
                    val_list.append(single_dataobject)
                data_list.append(val_list)
            else:
                single_dataobject = KGFlexDataObject(self.config, (train_val, test), self.side_information_data, self.args,
                                                              self.kwargs)
                data_list.append([single_dataobject])
        return data_list

    def generate_dataobjects_mock(self) -> t.List[object]:
        _column_names = ['userId', 'itemId', 'rating']
        training_set = np.hstack(
            (np.random.randint(0, 5 * 20, size=(5 * 20, 2)), np.random.randint(0, 2, size=(5 * 20, 1))))
        test_set = np.hstack(
            (np.random.randint(0, 5 * 20, size=(5 * 20, 2)), np.random.randint(0, 2, size=(5 * 20, 1))))

        side_information_data = SimpleNamespace()

        training_set = pd.DataFrame(np.array(training_set), columns=_column_names)
        test_set = pd.DataFrame(np.array(test_set), columns=_column_names)

        side_information_data.feature_map = {item: np.random.randint(0, 10, size=np.random.randint(0, 20)).tolist()
                                             for item in training_set['itemId'].unique()}

        data_list = [[KGFlexDataObject(self.config, (training_set, test_set), side_information_data,
                                                self.args, self.kwargs)]]

        return data_list

    def load_dataset_dataframe(self, file_ratings,
                               predicate_mapping_path,
                               item_features_path,
                               names=['userId', 'itemId', 'rating', 'timestamp']
                               ):
        # import dataset
        dataset = self.import_as_tsv(file_ratings, names=names, data_name='users data')

        # import predicate mapping
        predicate_mapping = self.import_as_tsv(predicate_mapping_path, names=['uri', 'predicate', 'predicate_order'])
        item_features = self.import_as_tsv(item_features_path, names=['itemId', 'predicate', 'object'])
        self.print_data_info(dataset)

        # ----- FILTER PREDICATES AND ITEMS -----
        item_features = self.filter_predicates_by_semantic("all", predicate_mapping, item_features)
        item_features = self.filter_predicates_by_frequency(item_features)
        dataset = self.data_cleaning(dataset, item_features)
        # export_dataset_as_tsv(self.dataset, output_folder=work_directory, file_name=f'dataset_{feature_type}')

        return dataset, item_features, predicate_mapping

    def import_as_tsv(self, path: str, names=None, index: str = None, data_name: str = ''):
        if names is None:
            names = ['userId', 'itemId', 'rating', 'timestamp']

        if index is not None:
            if index not in names:
                raise IndexError(f'specified index \'{index}\' not found in columns names.')

        data = pd.read_csv(path, sep='\t', names=names, index_col=index)
        if data_name == '':
            print(f'DATA IMPORT: dataset imported from {TextColor.OKGREEN}\'{path}\'{TextColor.ENDC}')
        else:
            print(f'DATA IMPORT: {data_name} dataset imported from {TextColor.OKGREEN}\'{path}\'{TextColor.ENDC}')

        return data

    def print_data_info(self, dataset: pd.DataFrame):
        """
        Prints the number of unique clients, unique items and rows present in the dataset
        :param dataset: pandas DataFrame
        :return: None
        """
        n_clients = len(dataset.userId.unique())
        n_items = len(dataset.itemId.unique())
        rows = len(dataset)

        print('\n************************************************\n')
        print(f'DATA INFO: Found {TextColor.BOLD}{n_clients}{TextColor.ENDC} different users in dataset')
        print(f'DATA INFO: Found {TextColor.BOLD}{n_items}{TextColor.ENDC} different items in dataset')
        print(f'DATA INFO: Found {TextColor.BOLD}{rows}{TextColor.ENDC} ratings in dataset')
        print('\n************************************************\n')

    def filter_predicates_by_semantic(self, feature_type="all", predicate_mapping={}, map={}):
        if feature_type == 'all':
            new_map = map.copy()
        else:
            if feature_type == 'categorical':
                categorical_regex = '^[^~]*terms\/subject(~.*terms\/subject)*(~.*skos\/core#broader)*$'
                accettable_predicates = predicate_mapping[
                    predicate_mapping['uri'].str.match(categorical_regex)].predicate.unique()
            new_map = map[map['predicate'].isin(accettable_predicates)]
        return new_map

    def filter_predicates_by_frequency(self, item_features):

        # ------------------------- DETECTING FEATURES WITH MISSING VALUES AND DISTINCT VALUES -------------------------


        # ------------ FILTERING ITEM FEATURES ------------
        # self.print_status('detecting and removing missing values and distinct values')

        '''
        # MISSING
        n_items = len(self.item_features.item_id.unique())
        items_per_predicate_dict = self.item_features.groupby(['predicate', 'item_id']).size() \
            .groupby('predicate').size().to_dict()
        items_per_predicate = np.array(tuple(items_per_predicate_dict.values()))
        pred_indexes = np.array(tuple(items_per_predicate_dict.keys()))
        missing_rate = (n_items - items_per_predicate) / n_items

        # DISTINCT
        objects_per_feature = self.item_features.groupby(['predicate', 'object']).size() \
            .groupby('predicate').size().to_numpy()
        occurrences_per_predicate_dict = self.item_features.groupby(['predicate']).size().to_dict()
        occurrences_per_predicate = np.array(tuple(items_per_predicate_dict.values()))

        assert np.array_equal(pred_indexes, np.array(tuple(occurrences_per_predicate_dict.keys())))

        distinct_rate = objects_per_feature / occurrences_per_predicate

        keep_mask = (missing_rate <= 0.997) * (distinct_rate <= 0.997)
        keep_set = set(pred_indexes[np.where(keep_mask)[0]])
        '''


        threshold = 10

        # SIMPLER VERSION
        occurrences_per_feature = item_features.groupby(['predicate', 'object']).size().to_dict()
        keep_set = {f for f in occurrences_per_feature if occurrences_per_feature[f] > threshold}

        print(f"PREDICATE FILTERING: kept {TextColor.BOLD}{len(keep_set)}{TextColor.ENDC}"
              f" FEATURES over {TextColor.BOLD}{len(item_features.predicate.unique())}{TextColor.ENDC}")

        new_map = item_features[
            item_features[['predicate', 'object']].set_index(['predicate', 'object']).index.map(
                lambda f: f in keep_set)]
        return new_map

    def data_cleaning(self, dataset, map):
        """
        Remove from self.dataset items that are not in filtered_items
        :return: None
        """
        # ------------------------- FILTERING ITEMS IN DATASET -------------------------

        filtered_items = map.itemId.unique()
        print(f"PREDICATE FILTERING: kept {TextColor.BOLD}{len(filtered_items)}{TextColor.ENDC}"
              f" ITEMS over {TextColor.BOLD}{len(dataset.itemId.unique())}{TextColor.ENDC}")

        new_dataset = dataset.set_index('itemId')
        # filter
        dataset_removed_items = set.difference(set(new_dataset.index.unique()), set(filtered_items))
        new_dataset = new_dataset.drop(list(dataset_removed_items))
        print('DATA INFO: {} items removed from training set'.format(len(dataset_removed_items)))
        # reindex
        new_dataset = new_dataset.reset_index().reindex(columns=['userId', 'itemId', 'rating', 'timestamp'])

        return new_dataset

    def print_status(self, message: str):
        print(f'{TextColor.BOLD}{TextColor.OKBLUE}'
              f'\nINFO: {message}...'
              f'{TextColor.ENDC}')


class KGFlexDataObject:
    """
    Load train and test dataset
    """

    def __init__(self, config, data_tuple, side_information_data, *args, **kwargs):
        self.logger = logging.get_logger(self.__class__.__name__, pylog.CRITICAL if config.config_test else pylog.DEBUG)
        self.config = config
        self.side_information_data = side_information_data
        self.args = args
        self.kwargs = kwargs
        self.train_dict = self.dataframe_to_dict(data_tuple[0])
        self.train_pd = data_tuple[0]

        self.users = list(self.train_dict.keys())
        self.num_users = len(self.users)
        self.items = list({k for a in self.train_dict.values() for k in a.keys()})
        self.num_items = len(self.items)

        # self.features = list({f for i in self.items for f in self.side_information_data.feature_map[i]})
        # self.factors = len(self.features)
        self.private_users = {p: u for p, u in enumerate(self.users)}
        self.public_users = {v: k for k, v in self.private_users.items()}
        self.private_items = {p: i for p, i in enumerate(self.items)}
        self.public_items = {v: k for k, v in self.private_items.items()}
        # self.private_features = {p: f for p, f in enumerate(self.features)}
        # self.public_features = {v: k for k, v in self.private_features.items()}
        self.transactions = sum(len(v) for v in self.train_dict.values())

        self.i_train_dict = {self.public_users[user]: {self.public_items[i]: v for i, v in items.items()}
                                for user, items in self.train_dict.items()}

        self.sp_i_train = self.build_sparse()
        self.sp_i_train_ratings = self.build_sparse_ratings()

        if len(data_tuple) == 2:
            self.test_dict = self.build_dict(data_tuple[1], self.users)
        else:
            self.val_dict = self.build_dict(data_tuple[1], self.users)
            self.test_dict = self.build_dict(data_tuple[2], self.users)


        # KaHFM compatible features

        kgflex_feature_df = self.side_information_data.feature_map.copy()

        def f(x):
            return str(x["predicate"]) + "><" + str(x["object"])

        kgflex_feature_df["bind"] = kgflex_feature_df.apply(f, axis=1)

        nitems = kgflex_feature_df["itemId"].nunique()
        threshold = 0.93
        kgflex_feature_df = kgflex_feature_df.groupby('bind').filter(lambda x: (1 - len(x) / nitems) <= threshold)
        print(f"Number of KaHFM features: {kgflex_feature_df['bind'].nunique()} with Threshold: {threshold}")

        feature_index = {k: p for p, k in enumerate(kgflex_feature_df["bind"].unique())}

        kgflex_feature_df["bind2"] = kgflex_feature_df["bind"].map(feature_index)
        kgflex_feature_df.drop(columns=["bind"], inplace=True)

        self.side_information_data.kahfm_feature_map = kgflex_feature_df.groupby("itemId")["bind2"].apply(list).to_dict()

        self.features = list(set(feature_index.values()))
        self.private_features = {p: f for p, f in enumerate(self.features)}
        self.public_features = {v: k for k, v in self.private_features.items()}

        if len(data_tuple) == 2:
            self.test_dict = self.build_dict(data_tuple[1], self.users)
            if hasattr(config, "negative_sampling"):
                val_neg_samples, test_neg_samples = NegativeSampler.sample(config, self.public_users, self.public_items, self.sp_i_train, None, self.test_dict)
                sp_i_test = self.to_bool_sparse(self.test_dict)
                test_candidate_items = test_neg_samples + sp_i_test
                self.test_mask = np.where((test_candidate_items.toarray() == True), True, False)
        else:
            self.val_dict = self.build_dict(data_tuple[1], self.users)
            self.test_dict = self.build_dict(data_tuple[2], self.users)
            if hasattr(config, "negative_sampling"):
                val_neg_samples, test_neg_samples = NegativeSampler.sample(config, self.public_users, self.public_items, self.sp_i_train, self.val_dict, self.test_dict)
                sp_i_val = self.to_bool_sparse(self.val_dict)
                sp_i_test = self.to_bool_sparse(self.test_dict)
                val_candidate_items = val_neg_samples + sp_i_val
                self.val_mask = np.where((val_candidate_items.toarray() == True), True, False)
                test_candidate_items = test_neg_samples + sp_i_test
                self.test_mask = np.where((test_candidate_items.toarray() == True), True, False)

        self.allunrated_mask = np.where((self.sp_i_train.toarray() == 0), True, False)

    def dataframe_to_dict(self, data):
        users = list(data['userId'].unique())

        "Conversion to Dictionary"
        ratings = {}
        for u in users:
            sel_ = data[data['userId'] == u]
            ratings[u] = dict(zip(sel_['itemId'], sel_['rating']))
        n_users = len(ratings.keys())
        n_items = len({k for a in ratings.values() for k in a.keys()})
        transactions = sum([len(a) for a in ratings.values()])
        sparsity = 1 - (transactions / (n_users * n_items))
        self.logger.info(f"Statistics\tUsers:\t{n_users}\tItems:\t{n_items}\tTransactions:\t{transactions}\t"
                         f"Sparsity:\t{sparsity}")
        return ratings

    def build_dict(self, dataframe, users):
        ratings = {}
        for u in users:
            sel_ = dataframe[dataframe['userId'] == u]
            ratings[u] = dict(zip(sel_['itemId'], sel_['rating']))
        return ratings

    def build_sparse(self):

        rows_cols = [(u, i) for u, items in self.i_train_dict.items() for i in items.keys()]
        rows = [u for u, _ in rows_cols]
        cols = [i for _, i in rows_cols]
        data = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
                             shape=(len(self.users), len(self.items)))
        return data

    def build_sparse_ratings(self):
        rows_cols_ratings = [(u, i, r) for u, items in self.i_train_dict.items() for i, r in items.items()]
        rows = [u for u, _, _ in rows_cols_ratings]
        cols = [i for _, i, _ in rows_cols_ratings]
        ratings = [r for _, _, r in rows_cols_ratings]

        data = sp.csr_matrix((ratings, (rows, cols)), dtype='float32',
                             shape=(len(self.users), len(self.items)))

        return data

    def get_test(self):
        return self.test_dict

    def get_validation(self):
        return self.val_dict if hasattr(self, 'val_dict') else None



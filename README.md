# KGFlex: a Knowledge-Aware Hybrid Recommender System

This is the official implementation of the paper:

*"Sparse Feature Factorization for Recommender Systems with Knowledge Graphs"*


## Description

## Table of Contents

- [Description](#description)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Reproduce Paper Results](#reproduce-paper-results)
  - [Experiments configuration](#experiments-configuration)
- [Datasets](#datasets)
- [KGFlex Parameters](#kgflex-parameters)

## Requirements 

This software works on the following operating systems:

* Linux
* Windows 10
* macOS X

Please, make sure to have the following installed on your system:

* Python 3.8.0 or later

KGFlex uses [Elliot](https://github.com/sisinflab/elliot) as reproducibility framework. This repository includes a ready-to-use distribution of Elliot including KGFlex and the other baselines analyzed in the paper. Thus, you can clone this repository and start to experiment with the models.

Finally, Python dependencies need to be installed with the command:

```
pip install -r requirements.txt
```

## Usage

Here we describe the steps to reproduce the results presented in the paper. Furthermore, we provide a description of how the experiments have been configured.

### Reproduce Paper Results

[Here](run.py) you can find a ready-to-run python file with all the pre-configured experiments cited in our paper.

You can easily run them with the following command:

```
python run.py
```

It trains our KGFlex model and the other baseline models with the three different datasets and, with one of them, also performs the semantic analysis.
A description of the datasets is provided [here](#datasets), while a comprehensive list of KGFlex parameters is available [here](#).

The results will be stored in the folder ```results/DATASET/```. Both the recommendation lists and the performance can be stored, depending on how the experiment is configured.

### Experiments configuration

The entry point of each experiment is the function ```run_experiment```, which accepts a configuration file that drives the whole experiment.
The configuration files can be found [here](config_files/).

In [run.py](run.py) all the experiments are executed sequencially, but it is also possibile to execute them separately, one by one.


Configuration files are ```YAML``` files within which all necessary information are provided to setup the experiment. An example of a KGFlex experiment configuration is shown below:

```
experiment:
  dataset: facebook-books
  data_config:
    strategy: dataset
    dataset_path: ../data/{0}/dataset.tsv
    dataloader: KGFlexLoader
    side_information:
      work_directory: ../data/{0}
      map: ../data/{0}/mapping.tsv
      features: ../data/{0}/item_features.tsv
      predicates: ../data/{0}/predicate_mapping.tsv
  prefiltering:
    strategy: iterative_k_core
    core: 5
  splitting:
    test_splitting:
        strategy: random_subsampling
        test_ratio: 0.2
  top_k: 10
  gpu: 1
  external_models_path: ../external/models/__init__.py
  evaluation:
    cutoffs: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    simple_metrics: [nDCGRendle2020, nDCG, HR, Precision, Recall, MAP, MRR, ItemCoverage, UserCoverage, NumRetrieved, UserCoverage, Gini, SEntropy, EFD, EPC]
  models:
    external.KGFlex:
      meta:
        verbose: True
        validation_rate: 10
        save_recs: True
      lr: [0.1, 0.01, 0.001]
      epochs: 100
      q: 0.1
      embedding: [1, 10, 100]
      parallel_ufm: 48
      first_order_limit: [0, 10, 100]
      second_order_limit: [0, 10, 100]
 ```

Each model requires specific parameters: a brief overview of KGFlex parameters is provided [here](#kgflex-parameters).

For further information about how to configure Elliot experiments, please refer to Elliot documentation.



## Datasets

Datasets can be found [here](data). Each folder contains the necessary to run the experiments.

Dataset | #Users | #Items | #Transactions | #Features
-- | -- | -- | -- | --
Facebook Books | 1398 | 2726 | 17626 | 306847
Yahoo Movies | 4000 |  2491 | 66600 | 1025399
Movielens 1M | 6040 | 3706 | 1000209 | 2284246

## KGFlex Parameters

The following are the parameters required by our KGFlex model:
- **lr**: learning rate, size of each learning step
- **epochs**: number of Gradient Descent iterations
- **q**: fraction of users selected for each learning epoch. It is a decimal number within 0 and 1. 
- **embedding**: item features embedding dimension
- **parallel_ufm**: number of parallel processes that will be executed during the user feature mapping operation.
- **first_order_limit**: max number of first order features for each user model
- **second_order_limit**: max number of second order features for each user model

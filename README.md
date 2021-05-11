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

## Requirements 

This software works on the following operating systems:

* Linux
* Windows 10
* macOS X

Please, make sure to have the following installed on your system:

* Python 3.8.0 or later
* [Elliot](https://github.com/sisinflab/elliot) (see how to install from source)

Then, to incorporate KGFlex in Elliot, copy and paste the KGFlex files into the Elliot home directory.

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

Configuration files are ```YAML``` files within which all necessary information are provided to setup the experiment: such as the dataset, the splitting strategy, the evaluation metrics and the models. A sigle experiment can contain more than one model and for each of them specific hyperparameters can be defined.

For further information about how to configure Elliot experiments, please refer to Elliot documentation.

### Datasets

Datasets can be found [here](data). Each folder contains the necessary to run the experiments.

Dataset | #Users | #Items | #Transactions | #Features
-- | -- | -- | -- | --
Facebook Books | 1398 | 2726 | 17626 | 306847
Yahoo Movies | 4000 |  2491 | 66600 | 1025399
Movielens 1M | 6040 | 3706 | 1000209 | 2284246


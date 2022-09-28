from elliot.run import run_experiment
import os

path = ["config_files_lgcn/yahoo_LGCN.yml",
        "config_files_lgcn/facebook-books_LGCN.yml",
        "config_files_lgcn/ml_LGCN.yml"]

for p in path:
    assert os.path.exists(p)

for p in path:
    run_experiment(p)

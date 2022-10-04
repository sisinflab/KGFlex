from elliot.run import run_experiment
import os

path = ["config_files_ncf/facebook-books_NCF.yml",
        "config_files_ncf/yahoo_NCF.yml",
        "config_files_ncf/ml_NCF.yml"]

for p in path:
    assert os.path.exists(p)

for p in path:
    run_experiment(p)

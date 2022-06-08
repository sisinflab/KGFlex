from elliot.run import run_experiment
import os

path = ["config_files/facebook-books_full_exploration.yml",
        "config_files/yahoo_full_exploration.yml",
        "config_files/ml_best_model.yml"]

for p in path:
    assert os.path.exists(p)

for p in path:
    run_experiment(p)

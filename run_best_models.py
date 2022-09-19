from elliot.run import run_experiment
import os

path = ["config_files/facebook-books_best_model.yml",
        "config_files/yahoo_books_best_model.yml",
        "config_files/ml_books_best_model.yml"]

for p in path:
    assert os.path.exists(p)

for p in path:
    run_experiment(p)

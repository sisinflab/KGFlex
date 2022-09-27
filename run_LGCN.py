from elliot.run import run_experiment
import os

path = ["config_files/yahoo_LGCN.yml"]

for p in path:
    assert os.path.exists(p)

for p in path:
    run_experiment(p)

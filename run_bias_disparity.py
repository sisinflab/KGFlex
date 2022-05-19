from elliot.run import run_experiment
import os


# BEFORE RUNNING THIS SCRIPT PLEASE READ THE CONFIGURATION FILE BELOW
# READ ME -> config_files/bias_disparity.yml
path = ['config_files/bias_disparity.yml']

for p in path:
    assert os.path.exists(p)

for p in path:
    run_experiment(p)

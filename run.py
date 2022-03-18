from elliot.run import run_experiment
import os

path = ["config_files/yahoo-movies_kgflex_npr_expsol100_npr2.yml",
        "config_files/yahoo-movies_kgflex_npr_expsol200_npr2.yml",
        "config_files/yahoo-movies_kgflex_npr_expsol100_npr20.yml",
        "config_files/yahoo-movies_kgflex_npr_expsol200_npr20.yml"]

for p in path:
    assert os.path.exists(p)


for p in path:
    run_experiment(p)

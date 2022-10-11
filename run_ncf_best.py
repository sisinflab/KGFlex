from elliot.run import run_experiment
import os

path = ["config_lgcn_ncf_bestmodels/facebook-books_NCF.yml",
        "config_lgcn_ncf_bestmodels/ml_NCF.yml",
        "config_lgcn_ncf_bestmodels/yahoo_NCF.yml"]

path = ["config_lgcn_ncf_bestmodels/yahoo_NCF.yml"]

for p in path:
    assert os.path.exists(p)

for p in path:
    run_experiment(p)

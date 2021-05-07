from elliot.run import run_experiment

run_experiment("config_files/facebook-books_kgflex.yml")
run_experiment("config_files/facebook-books_baselines.yml")
run_experiment("config_files/yahoo-movies_kgflex.yml")
run_experiment("config_files/yahoo-movies_baselines.yml")
run_experiment("config_files/movielens1m_kgflex.yml")
run_experiment("config_files/movielens1m_baselines.yml")

run_experiment("config_files/yahoo-movies_semantics-analysis.yml")

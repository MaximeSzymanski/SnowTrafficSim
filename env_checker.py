from pettingzoo.test import parallel_api_test
from model.env_wrapper import MontrealSnowEnv
from pettingzoo.test import seed_test, parallel_seed_test
from gymnasium.utils.env_checker import check_env


env = MontrealSnowEnv(graph_filepath="data/plateau_mont_royal_drive.graphml", num_blowers=1, num_trucks=1)

parallel_api_test(env, num_cycles=1000)
#check_env(env)
parallel_seed_test(MontrealSnowEnv, num_cycles=1000)
import os, ray, shutil
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from model.env_wrapper import MontrealSnowEnv
from nn.gnn_policy import MontrealGNNModel

ModelCatalog.register_custom_model("montreal_gnn", MontrealGNNModel)

def env_creator(config):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(base_dir, "data", "plateau_mont_royal_drive.graphml")
    env = MontrealSnowEnv(graph_filepath=graph_path, num_blowers=2, num_trucks=4)
    return ParallelPettingZooEnv(env)

if __name__ == "__main__":
 
        
    ray.shutdown()

    
    ## 3. Simple Init - removed the 'unexpected' config parameter
    ray.init(
        num_cpus=4,
        include_dashboard=False,
        # We only keep the timeout extension to help the disk wake up
        _system_config={
            "gcs_server_request_timeout_seconds": 60,
        }
    )

    register_env("montreal_snow_v1", lambda config: env_creator(config))

    temp_env = env_creator({})
    obs_space_blower = temp_env.observation_space["blower_0"]
    act_space_blower = temp_env.action_space["blower_0"]
    obs_space_truck = temp_env.observation_space["truck_0"]
    act_space_truck = temp_env.action_space["truck_0"]

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment("montreal_snow_v1")
        .framework("torch")
        .env_runners(num_env_runners=1) 
        .training(
            model={"custom_model": "montreal_gnn"},
            lr=5e-5,
            train_batch_size=512
        )
        .multi_agent(
            policies={
                "blower_policy": (None, obs_space_blower, act_space_blower, {}),
                "truck_policy": (None, obs_space_truck, act_space_truck, {}),
            },
            policy_mapping_fn=lambda aid, *args, **kw: "blower_policy" if "blower" in aid else "truck_policy",
        )
    )

    tune.run(
        "PPO", 
        config=config.to_dict(), 
        stop={"timesteps_total": 100000},
        # Auto-delete logs if they get too big
        checkpoint_config=tune.CheckpointConfig(checkpoint_frequency=0) 
    )
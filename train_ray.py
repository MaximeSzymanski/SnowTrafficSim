import os, ray, shutil
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks

import networkx as nx
import numpy as np

# Import your custom modules
from model.env_wrapper import MontrealSnowEnv
from nn.gnn_policy import MontrealGNNModel 
from nn.linear_policy import MontrealLinearModel

ModelCatalog.register_custom_model("montreal_linear", MontrealLinearModel)
ModelCatalog.register_custom_model("montreal_gnn", MontrealGNNModel)

print("Generating Adjacency Matrix for GNN...")
base_dir = os.path.dirname(os.path.abspath(__file__))
graph_path = os.path.join(base_dir, "data", "plateau_mont_royal_drive.graphml")
temp_graph = nx.read_graphml(graph_path)

# Creates a grid of 1s and 0s showing street connections
adj_matrix = nx.to_numpy_array(temp_graph, weight=None).tolist()

os.makedirs("result", exist_ok=True)

# --- MODERN CURRICULUM & METRICS CALLBACK ---
class SnowMetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index: int, **kwargs):
        # PettingZoo copies the global info dict to all agents. 
        # We grab the info from "blower_0" to get the episode stats.
        info = episode.last_info_for("blower_0")
        
        if info:
            if "snow_removed" in info:
                episode.custom_metrics["snow_removed"] = info["snow_removed"]
            if "time_elapsed_mins" in info:
                episode.custom_metrics["time_elapsed_mins"] = info["time_elapsed_mins"]

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        """
        Dynamically updates the environment difficulty based on total timesteps.
        """
        timesteps = result.get("timesteps_total", 0)
        
        # 1. Determine Curriculum Level
        if timesteps < 1000000:
            level = 1 # Phase 1: Buddy System
        elif timesteps < 2500000:
            level = 2 # Phase 2: Neighborhood Radius
        else:
            level = 3 # Phase 3: Full Random City
            
        # 2. Push the level to all remote environment workers securely
        algorithm.workers.foreach_worker(
            lambda worker: worker.foreach_env(
                lambda env: env.unwrapped.set_task(level) if hasattr(env.unwrapped, "set_task") else None
            )
        )
        
        # 3. Log the level directly to TensorBoard so we can track it
        if "custom_metrics" not in result:
            result["custom_metrics"] = {}
        result["custom_metrics"]["curriculum_level_mean"] = level


def env_creator(config):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(base_dir, "data", "plateau_mont_royal_drive.graphml")
    
    render_mode = config.get("render_mode", None)
    print(f"Creating env with render_mode: {render_mode}") 
    
    env = MontrealSnowEnv(graph_filepath=graph_path, num_blowers=1, num_trucks=2)
    env.render_mode = render_mode
    return ParallelPettingZooEnv(env)


if __name__ == "__main__":
    ray_tmp = "/tmp/r"
    if os.path.exists(ray_tmp): shutil.rmtree(ray_tmp)
    os.makedirs(ray_tmp, exist_ok=True)
    
    ray.shutdown()
    ray.init(num_cpus=4, _temp_dir=ray_tmp, _system_config={"gcs_server_request_timeout_seconds": 60})

    register_env("montreal_snow_v1", lambda config: env_creator(config))
    
    # Dummy env to get spaces
    temp_env = env_creator({})
    obs_space_blower = temp_env.observation_space["blower_0"]
    act_space_blower = temp_env.action_space["blower_0"]
    obs_space_truck = temp_env.observation_space["truck_0"]
    act_space_truck = temp_env.action_space["truck_0"]

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(
            "montreal_snow_v1", 
            env_config={"render_mode": None}
            # Notice we removed env_task_fn here entirely!
        )
        .framework("torch")
        .callbacks(SnowMetricsCallback) # <--- Hooking in our custom callback!
        .training(
            model={"custom_model": "montreal_gnn",
                  "custom_model_config": {"adj_matrix": adj_matrix}},
            lr=1e-3,
            train_batch_size=8192,
            minibatch_size=512,
        )
        .env_runners(
            num_env_runners=2,
            num_envs_per_env_runner=4 
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
        stop={"timesteps_total": 5000000},
        checkpoint_config=tune.CheckpointConfig(checkpoint_frequency=10),
        storage_path=os.path.expanduser("~/ray_results/snow_removal")
    )
"""
Main execution script for distributed Multi-Agent Reinforcement Learning (MARL) 
training of the Montreal Snow Removal fleet. 

Utilizes Ray RLlib for PPO optimization, custom Graph Neural Network (GNN) policies, 
and automated curriculum learning. Includes a custom callback for rendering 
evaluation videos during the training lifecycle.
"""

import os
import shutil
import imageio
import numpy as np
import networkx as nx

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# Set dummy video driver for headless rendering on macOS/Linux servers
os.environ["SDL_VIDEODRIVER"] = "dummy" 

from model.env_wrapper import MontrealSnowEnv
from nn.gnn_policy import MontrealGNNModel 
from nn.linear_policy import MontrealLinearModel

ModelCatalog.register_custom_model("montreal_linear", MontrealLinearModel)
ModelCatalog.register_custom_model("montreal_gnn", MontrealGNNModel)

# Precompute the adjacency matrix for the GNN policy
print("Generating Adjacency Matrix for GNN...")
base_dir = os.path.dirname(os.path.abspath(__file__))
graph_path = os.path.join(base_dir, "data", "plateau_mont_royal_drive.graphml")
temp_graph = nx.read_graphml(graph_path)
adj_matrix = nx.to_numpy_array(temp_graph, weight=None).tolist()

os.makedirs(os.path.join(base_dir, "result"), exist_ok=True)


class SnowMetricsCallback(DefaultCallbacks):
    """
    Custom RLlib callbacks to log domain-specific metrics, advance the 
    curriculum difficulty, and periodically render evaluation videos.
    """

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index: int, **kwargs):
        """
        Extracts custom environment information at the end of an episode and 
        appends it to RLlib's custom metrics tracker.
        """
        info = episode.last_info_for("blower_0")
        if info:
            if "snow_removed" in info:
                episode.custom_metrics["snow_removed"] = info["snow_removed"]
            if "time_elapsed_mins" in info:
                episode.custom_metrics["time_elapsed_mins"] = info["time_elapsed_mins"]

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        """
        Executes post-iteration logic: updates the environment's curriculum level 
        based on total timesteps and renders a deterministic simulation video 
        every 10 training iterations.
        """
        timesteps = result.get("timesteps_total", 0)
        iteration = result.get("training_iteration", 0)
        
        # 1. Curriculum Advancement
        if timesteps < 1000000: 
            level = 1
        elif timesteps < 2500000: 
            level = 2
        else: 
            level = 3
            
        algorithm.env_runner_group.foreach_env_runner(
            lambda worker: worker.foreach_env(
                lambda env: env.par_env.set_task(level) if hasattr(env, "par_env") else None
            )
        )
        
        if "custom_metrics" not in result: 
            result["custom_metrics"] = {}
        result["custom_metrics"]["curriculum_level_mean"] = level

        # 2. Automated Video Evaluation
        if iteration > 0 and iteration % 10 == 0:
            print(f"🎬 Filming evaluation video for iteration {iteration}...")
            eval_env = env_creator({"render_mode": "rgb_array"})
            writer = None
            
            try:
                if hasattr(eval_env, "par_env"):
                    eval_env.par_env.set_task(level)
                
                obs, info = eval_env.reset()
                
                project_dir = os.path.join(base_dir, "result")
                if not os.path.exists(project_dir):
                    os.makedirs(project_dir, exist_ok=True)
                    
                video_path = os.path.join(project_dir, f"eval_video_iter_{iteration}.mp4")
                writer = imageio.get_writer(video_path, fps=10)
                
                frame = eval_env.par_env.render()
                if frame is not None:
                    writer.append_data(frame)
                
                done = False
                max_steps = 1000 
                steps = 0
                
                while not done and steps < max_steps:
                    actions = {}
                    for agent_id, agent_obs in obs.items():
                        policy_id = "blower_policy" if "blower" in agent_id else "truck_policy"
                        actions[agent_id] = algorithm.compute_single_action(
                            observation=agent_obs,
                            policy_id=policy_id,
                            explore=False 
                        )
                    
                    obs, rewards, terminated, truncated, infos = eval_env.step(actions)
                    
                    frame = eval_env.par_env.render()
                    if frame is not None:
                        writer.append_data(frame)
                    
                    done = all(terminated.values()) or all(truncated.values())
                    steps += 1
                  
                print(f"✅ Video successfully saved: {video_path}")

            except Exception as e:
                print(f"⚠️ Failed to generate video: {e}")
                
            finally:
                if writer is not None:
                    writer.close()
                if eval_env is not None:
                    eval_env.close()


def env_creator(config):
    """
    Instantiates the PettingZoo environment with the specified configuration 
    and graph topology.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_path = os.path.join(base_dir, "data", "plateau_mont_royal_drive.graphml")
    
    render_mode = config.get("render_mode", None)
    
    env = MontrealSnowEnv(graph_filepath=graph_path, num_blowers=3, num_trucks=10)
    env.render_mode = render_mode
    return ParallelPettingZooEnv(env)


if __name__ == "__main__":
    ray_tmp = "/tmp/r"
    if os.path.exists(ray_tmp): 
        shutil.rmtree(ray_tmp)
    os.makedirs(ray_tmp, exist_ok=True)
    
    ray.shutdown()
    ray.init(num_cpus=4, _temp_dir=ray_tmp, _system_config={"gcs_server_request_timeout_seconds": 60})

    register_env("montreal_snow_v1", lambda config: env_creator(config))
    
    # Extract observation and action spaces for policy definitions
    temp_env = env_creator({})
    obs_space_blower = temp_env.observation_space["blower_0"]
    act_space_blower = temp_env.action_space["blower_0"]
    obs_space_truck = temp_env.observation_space["truck_0"]
    act_space_truck = temp_env.action_space["truck_0"]
    temp_env.close() 

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(
            "montreal_snow_v1", 
            env_config={"render_mode": None}
        )
        .framework("torch")
        .callbacks(SnowMetricsCallback) 
        .training(
            model={
                "custom_model": "montreal_gnn",
                "custom_model_config": {"adj_matrix": adj_matrix}
            },
            lr=1e-3,
            train_batch_size=8192,
            minibatch_size=512,
        )
        .env_runners(
            num_env_runners=3,
            num_envs_per_env_runner=4,
        ) 
        .multi_agent(
            policies={
                "blower_policy": (None, obs_space_blower, act_space_blower, {}),
                "truck_policy": (None, obs_space_truck, act_space_truck, {}),
            },
            policy_mapping_fn=lambda aid, *args, **kw: "blower_policy" if "blower" in aid else "truck_policy",
            count_steps_by="agent_steps"
        )
    )

    tune.run(
        "PPO", 
        config=config.to_dict(), 
        stop={"timesteps_total": 5_000_000},
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=1,
            checkpoint_at_end=True,
            num_to_keep=3
        ),
        storage_path=os.path.expanduser("~/ray_results/snow_removal")
    )
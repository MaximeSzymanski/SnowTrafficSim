import os
import time
import numpy as np
import imageio # ADD THIS IMPORT

from model.env_wrapper import MontrealSnowEnv 

def run_local_eval(episodes=1):
    print("❄️  Initializing Local Montreal Snow Simulator ❄️")
    
    env = MontrealSnowEnv(
        graph_filepath="data/plateau_mont_royal_drive.graphml", 
        num_blowers=2, 
        num_trucks=6,
        render_mode="rgb_array"
    )
    
    os.makedirs("videos", exist_ok=True) # Ensure video folder exists
    
    for ep in range(episodes):
        print(f"\n--- Starting Episode {ep + 1} ---")
        observations, infos = env.reset()
        
        step_count = 0
        total_rewards = {agent: 0.0 for agent in env.possible_agents}
        frames = [] # 1. Initialize empty video frame list
        
        # Capture the starting frame
        frames.append(env.render()) 
        
        while env.agents:
            actions = {}
            for agent in env.agents:
                action_mask = observations[agent]["action_mask"]
                valid_actions = np.where(action_mask == 1)[0]
                if len(valid_actions) > 0:
                    actions[agent] = np.random.choice(valid_actions)
                else:
                    actions[agent] = 0
            
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # 2. Capture the environment state as an image
            frames.append(env.render())
            
            step_count += 1
            for agent in env.agents:
                if agent in rewards:
                    total_rewards[agent] += rewards[agent]
            
            observations = next_observations
            
            if step_count % 10 == 0:
                current_snow = getattr(env.sim, 'total_snow', 0)
                b_reward = total_rewards.get('blower_0', 0)
                t_reward = total_rewards.get('truck_0', 0)
                print(f"Step: {step_count:03d} | Snow Remaining: {current_snow:8.1f} | Blower_0 Rew: {b_reward:6.1f} | Truck_0 Rew: {t_reward:6.1f}")
        
        print(f"\n✅ Episode {ep + 1} Complete!")
        print(f"Total Steps: {step_count}")
        print(f"Final Snow Remaining: {getattr(env.sim, 'total_snow', 0):.1f}")
        
        # 3. Stitch the frames into an MP4
        video_path = f"videos/local_eval_ep{ep + 1}.mp4"
        print(f"🎥 Saving video to {video_path}...")
        imageio.mimsave(video_path, frames, fps=15)
        print("Done!")
        
if __name__ == "__main__":
    run_local_eval(episodes=1)
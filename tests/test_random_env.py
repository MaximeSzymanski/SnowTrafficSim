import random
import numpy as np
from model.env_wrapper import MontrealSnowEnv

def run_random_test(graph_file, num_blowers=2, num_trucks=6, max_steps=50):
    print("Initializing Montreal Snow Environment...")
    env = MontrealSnowEnv(graph_filepath=graph_file, num_blowers=num_blowers, num_trucks=num_trucks)
    
    # 1. Reset the environment to get the initial state
    observations, infos = env.reset()
    
    step_count = 0
    total_rewards = {agent: 0.0 for agent in env.agents}

    print("\nStarting Random Action Loop...")
    
    # Run until all agents terminate or we hit the max step limit
    while env.agents and step_count < max_steps:
        actions = {}
        
        # 2. Select actions based strictly on the Action Mask
        for agent in env.agents:
            # Extract the mask we built in _get_graph_state
            mask = observations[agent]["action_mask"]
            
            # Find the indices where the mask is 1 (Valid Actions)
            valid_action_indices = np.where(mask == 1)[0]
            
            if len(valid_action_indices) > 0:
                # Pick a random action strictly from the valid choices
                chosen_action = random.choice(valid_action_indices)
            else:
                # Failsafe: 4 (Idle) for blowers, 0 (Follow Blower 0) for trucks
                chosen_action = 4 if "blower" in agent else 0 
                
            actions[agent] = chosen_action

        # 3. Step the environment forward
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Track cumulative rewards
        for agent in env.agents:
            total_rewards[agent] += rewards.get(agent, 0.0)

        # 4. Print Debugging Info
        print(f"\n--- Step {step_count} | Sim Time: {env.sim.env.now}s ---")
        
        # Print a sample Blower and Truck to keep the terminal readable
        sample_blower = "blower_0"
        sample_truck = "truck_0"
        
        if sample_blower in env.agents:
            b_act = actions[sample_blower]
            b_rew = rewards[sample_blower]
            print(f"[{sample_blower}] Action: {b_act} | Reward this step: {b_rew:+.2f} | Total: {total_rewards[sample_blower]:+.2f}")
            
        if sample_truck in env.agents:
            t_act = actions[sample_truck]
            t_rew = rewards[sample_truck]
            print(f"[{sample_truck}] Action: {t_act} | Reward this step: {t_rew:+.2f} | Total: {total_rewards[sample_truck]:+.2f}")

        step_count += 1

    print("\n--- Simulation Complete ---")
    print("Final Cumulative Rewards:")
    for agent, reward in total_rewards.items():
        print(f"  {agent}: {reward:.2f}")

if __name__ == "__main__":
    # Point this to your saved Montreal OpenStreetMap file
    graph_file = "data/plateau_mont_royal_drive.graphml"
    
    # Run a short 30-step test with a small fleet
    run_random_test(graph_file, num_blowers=2, num_trucks=4, max_steps=30)
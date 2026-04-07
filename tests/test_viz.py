import random
import numpy as np
import matplotlib.pyplot as plt
from model.env_wrapper import MontrealSnowEnv

def run_visual_simulation():
    graph_file = "data/plateau_mont_royal_drive.graphml"
    
    # Use a smaller fleet for better visibility initially
    env = MontrealSnowEnv(graph_filepath=graph_file, num_blowers=20, num_trucks=40)
    observations, infos = env.reset()
    
    print("Visual Simulation Starting... Close the window to stop.")
    
    try:
        for step in range(200):
            actions = {}
            for agent in env.agents:
                mask = observations[agent]["action_mask"]
                valid_actions = np.where(mask == 1)[0]
                actions[agent] = random.choice(valid_actions)

            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # THE MAGIC LINE
            env.render()

            if all(terminations.values()):
                break
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    run_visual_simulation()
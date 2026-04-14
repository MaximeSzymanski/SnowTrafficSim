# Montreal Snow Removal: Multi-Agent RL Simulation

A custom Multi-Agent Reinforcement Learning (MARL) environment and training pipeline designed to optimize city-scale snow removal logistics.

This project simulates a collaborative fleet of snowblowers and dump trucks navigating a real-world street graph of the Plateau Mont-Royal in Montreal. The agents are trained using Proximal Policy Optimization (PPO) and a custom Graph Neural Network (GNN) architecture to handle spatial reasoning and dynamic dispatching.

![GifResult](gif/result.gif)

## The problem

Snow removal in dense urban environments requires tight coordination between heterogeneous fleets. Snowblowers must carve continuous paths through snow-laden streets, but they cannot store snow—they require an empty dump truck driving alongside them. Once full, dump trucks must independently route to the nearest depot, empty their payload, and dynamically reassign themselves to active blowers.

## Tech Stack
- **Reinforcement Learning**: Ray / RLlib (PPO)

- **Neural Network**: PyTorch (Custom Graph Convolutional Network)

- **Environment API**: PettingZoo (ParallelEnv)

- **Simulation Engine**: SimPy (Event-driven concurrency)

- **Graph Mathematics**: NetworkX

- **Rendering**: PyGame / ImageIO

## Architecture

**1. The Environment** (env_wrapper.py & sim_engine.py)
Built from scratch using PettingZoo, the environment turns a `.graphml` file of Montreal into a bidirectional NetworkX directed graph. It handles:

- **Physics & Routing**: SimPy manages continuous time, while a globally cached shortest-path algorithm prevents Dijkstra-calculation bottlenecks during training.

- **Heterogeneous Action Spaces**: * _Blowers (Discrete 5)_: Tactically steer through intersections (North, South, East, West, Wait) using angle-based action masking.

     - _Trucks (Discrete N)_: Act as dispatchers, selecting global targets (Blower 1, Blower 2, Depot A, Depot B) and relying on the routing engine to drive.


**2. The Brain** (gnn_policy.py)
Because the map topology is complex and agents need to "see" traffic blocks away, the policy utilizes a Graph Neural Network (GNN).

- **Sparse Matrix Multiplication**: To prevent OOM errors with large batch sizes, the adjacency matrix is normalized and converted to a PyTorch Sparse Tensor.

- **Message Passing**: Two layers of Graph Convolutions allow intersections to average their traffic and snow data with their neighbors, giving every agent a 2-block receptive field.

- **Dynamic Slicing**: The Actor-Critic network concatenates a macro-level average of the entire city with a micro-level slice of the specific node the agent is currently standing on.

## Key Engineering Challenges Solved

- **The Cul-de-Sac Deadlock**: Real-world maps contain one-way dead-ends. Blower action masks were dynamically updated to detect `0` forward successors, invert the street angle math, and execute forced 180-degree U-turns.

- **Ghost Assignments**: In early iterations, full dump trucks would leave for the depot without breaking their memory link to the blower, causing the blower to wait forever. The environment now strictly manages contract severing during state transitions.

- **The "Hot Potato" Fuel Penalty**: Trucks initially converged on a single depot regardless of distance. A dynamic fuel penalty was implemented `(payload / max_capacity) * 0.5`, mathematically forcing the neural network to optimize for the absolute closest drop-off point to save its score.

- **Dithering & Loop Traps**: To prevent untrained agents from driving in safe 3-block circles to run out the clock, a short-term node history memory was added. Detecting a loop triggers an immediate `-100.0` penalty and episode truncation.

## Installation & Usage

**1. Clone the repository and install dependencies**:
```bash
git clone https://github.com/yourusername/montreal-snow-marl.git
cd montreal-snow-marl
pip install ray[rllib] torch networkx pettingzoo pygame imageio simpy
```

**2. Run the training pipeline**:
```bash
python -m train_ray
```

*Note: The script includes an Auto-Director callback that seamlessly renders evaluation videos (`.mp4`) into the `/result` folder every 10 iterations without interrupting the distributed training.*


## Current Status & Future Work
This project is an active Work-in-Progress. Current focus areas include:

- **Hyperparameter Tuning**: Balancing the GNN learning rate against the PPO entropy coefficient to encourage further exploration of the map boundaries.

- **Curriculum Expansion**: Implementing dynamic weather events (snow falling during the episode rather than just at initialization).

- **Traffic Simulation**: Adding civilian traffic entities that act as dynamic obstacles, forcing the snow removal fleet to reroute in real-time.

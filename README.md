# 🚜 AI That Learns to Clear a City  
### Multi-Agent Reinforcement Learning for Montreal Snow Removal

This project implements a **custom multi-agent reinforcement learning system** designed to simulate and optimize snow removal operations on a **real Montreal road network**.

It focuses on building a **realistic, scalable MARL environment** where heterogeneous agents must coordinate under physical and logistical constraints.

🎬 *Simulation demo*  
![GifResult](gif/result.gif)

---

## 💡 Project Goal

Snow removal in dense cities is a complex coordination problem:

- Snowblowers clear streets but cannot store snow  
- Dump trucks must follow, collect, and unload at depots  
- Poor coordination leads to idle time, deadlocks, and inefficiency  

This project explores how **Multi-Agent RL + Graph Neural Networks** can learn to solve this problem.

---

## 🧠 What This Project Demonstrates

- Building a **custom MARL environment from scratch**
- Designing **heterogeneous agent systems**
- Applying **Graph Neural Networks to spatial problems**
- Handling **real-world constraints (routing, capacity, timing)**
- Engineering **scalable RL training pipelines**

---

## 🏗️ Architecture Overview

### 1. Environment (PettingZoo + SimPy)

A fully custom environment simulating snow removal dynamics:

- Real-world **road graph** from Montreal (.graphml)
- **Event-driven simulation** using SimPy
- Multi-agent system:
  - **SnowBlowers** → clear snow from edges  
  - **DumpTrucks** → collect and unload snow  

#### Action Spaces
- **Blowers (Discrete 5)**: move (N/S/E/W) or wait  
- **Trucks (Discrete N)**: assign to blower or depot  

---

### 2. Model (Graph Neural Network)

A custom GNN processes the entire road network:

- Sparse adjacency matrix for efficiency  
- Multi-layer message passing  
- Combines:
  - Global map embedding  
  - Local node features  
  - Agent-specific state  

This allows agents to reason about:
- Snow distribution  
- Nearby agents  
- Road connectivity  

---

### 3. Training Pipeline (RLlib PPO)

- Distributed training with Ray RLlib  
- Multi-policy setup:
  - blower_policy  
  - truck_policy  
- Custom callback system:
  - Curriculum learning  
  - Metric tracking  
  - Automated video rendering  

---

## ⚙️ Key Engineering Challenges Solved

### 🚧 Real-World Graph Constraints
- One-way streets and dead ends  
- Dynamic action masking to ensure valid moves  

---

### 🔗 Multi-Agent Coordination Bugs
- Fixed “ghost assignments” between trucks and blowers  
- Enforced consistent state transitions  

---

### 🔁 Looping & Degenerate Policies
- Added loop detection via node history  
- Penalized repetitive behavior  

---

### ⚡ Performance & Scalability
- Cached shortest paths (avoids repeated Dijkstra)  
- Sparse tensor operations for GNN  
→ Enables training on large graphs  

---

## 🚀 Installation

Clone the repository and install dependencies:

git clone https://github.com/yourusername/montreal-snow-marl.git  
cd montreal-snow-marl  

pip install ray[rllib] torch networkx pettingzoo pygame imageio simpy  

---

## ▶️ Usage

Run training:

python train.py  

Outputs:
- Checkpoints → ~/ray_results/snow_removal  
- Evaluation videos → /result  

---

## 🎬 Automated Evaluation

The training loop includes a custom callback that:

- Runs deterministic simulations during training  
- Records .mp4 videos of agent behavior  

This enables **visual inspection of learned policies**, which is critical in MARL.

---

## 🔮 Next Steps

- Train agents to convergence and benchmark performance  
- Add dynamic snowfall during episodes  
- Introduce traffic as dynamic obstacles  
- Improve reward shaping for global coordination  

---

## 📌 Summary

This project focuses on **engineering a realistic MARL system**, including:

- A custom environment with real-world constraints  
- A scalable GNN-based policy  
- A distributed RL training pipeline  

It serves as a foundation for experimenting with **multi-agent coordination at city scale**.

---

⭐ If you find this interesting, feel free to star the repo or reach out!

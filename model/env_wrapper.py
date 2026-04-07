import functools
import math
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from pettingzoo import ParallelEnv
from gymnasium import spaces

from model.sim_engine import SnowRemovalSim, SnowBlower, DumpTruck 

class MontrealSnowEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "montreal_snow_v1"}

    def __init__(self, graph_filepath, num_blowers=2, num_trucks=6):
        self.graph_filepath = graph_filepath
        self.num_blowers = num_blowers
        self.num_trucks = num_trucks
        self.num_dumps = 2
        
        self.possible_agents = (
            [f"blower_{i}" for i in range(num_blowers)] +
            [f"truck_{i}" for i in range(num_trucks)]
        )
        self.agents = []
        self.sim = None 

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        graph_obs_space = spaces.Dict({
            "intersections": spaces.Box(low=0, high=1, shape=(20000, 1), dtype=np.float32), 
            "blowers": spaces.Box(low=-1, high=20000, shape=(self.num_blowers, 2), dtype=np.float32),
            "trucks": spaces.Box(low=-1, high=20000, shape=(self.num_trucks, 3), dtype=np.float32),
            "edges": spaces.Box(low=0, high=100, shape=(50000, 1), dtype=np.float32)
        })
        num_act = 5 if "blower" in agent else self.num_blowers + self.num_dumps
        return spaces.Dict({
            "observation": graph_obs_space,
            "action_mask": spaces.Box(low=0, high=1, shape=(num_act,), dtype=np.int8)
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        num_act = 5 if "blower" in agent else self.num_blowers + self.num_dumps
        return spaces.Discrete(num_act)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.sim = SnowRemovalSim(self.graph_filepath)
        self._spawn_fleet()
        return self._get_graph_state(), {a: {} for a in self.possible_agents}

    def _spawn_fleet(self):
        import random
        for i in range(self.num_blowers):
            start_node = random.choice(self.sim.nodes)
            self.sim.blowers.append(SnowBlower(self.sim.env, self.sim, f"blower_{i}", start_node))
        for i in range(self.num_trucks):
            start_node = random.choice(self.sim.nodes)
            self.sim.trucks.append(DumpTruck(self.sim.env, self.sim, f"truck_{i}", start_node))

    def step(self, actions):
        for agent_id, action in actions.items():
            if agent_id in self.agents:
                if "blower" in agent_id: self._apply_blower_action(agent_id, int(action))
                else: self._apply_truck_action(agent_id, int(action))

        self.sim.env.run(until=self.sim.env.now + 10)

        observations = self._get_graph_state()
        rewards = self._calculate_rewards()
        
        is_done = self.sim.env.now > (60 * 20)
        
        # Standard PettingZoo: Terminate everyone at once
        terminations = {a: is_done for a in self.possible_agents}
        truncations = {a: is_done for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}

        if is_done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _get_graph_state(self):
        # 1. Intersections
        int_feat = np.zeros((20000, 1), dtype=np.float32)
        for i, node in enumerate(self.sim.nodes[:20000]):
            if self.sim.intersections[node].count > 0: int_feat[i][0] = 1.0

        # 2. Blowers
        b_feat = np.zeros((self.num_blowers, 2), dtype=np.float32)
        for i, b in enumerate(self.sim.blowers):
            b_feat[i][0] = float(self.sim.nodes.index(b.current_node))
            b_feat[i][1] = 1.0 if getattr(b, 'is_waiting', False) else 0.0

        # 3. Trucks
        t_feat = np.zeros((self.num_trucks, 3), dtype=np.float32)
        for i, t in enumerate(self.sim.trucks):
            t_feat[i][0] = float(self.sim.nodes.index(t.current_node))
            t_feat[i][1] = float(t.payload / t.max_capacity)
            t_feat[i][2] = float(self.sim.blowers.index(t.blower)) if t.blower else -1.0

        # 4. Edges
        e_feat = np.zeros((50000, 1), dtype=np.float32)
        for i, (u, v, k, d) in enumerate(list(self.sim.graph.edges(keys=True, data=True))[:50000]):
            e_feat[i][0] = float(d.get('snow_depth', 0.0))

        base = {"intersections": int_feat, "blowers": b_feat, "trucks": t_feat, "edges": e_feat}
        return {a: {"observation": base, "action_mask": self._get_mask(a)} for a in self.possible_agents}

    def _get_mask(self, agent_id):
        if "blower" in agent_id:
            idx = int(agent_id.split("_")[1]); blower = self.sim.blowers[idx]
            mask = np.zeros(5, dtype=np.int8); mask[4] = 1
            cx, cy = float(self.sim.graph.nodes[blower.current_node]['x']), float(self.sim.graph.nodes[blower.current_node]['y'])
            for n in self.sim.graph.successors(blower.current_node):
                nx, ny = float(self.sim.graph.nodes[n]['x']), float(self.sim.graph.nodes[n]['y'])
                angle = (math.degrees(math.atan2(nx - cx, ny - cy)) + 360) % 360
                if 315 <= angle or angle <= 45: mask[0] = 1
                elif 45 < angle <= 135: mask[2] = 1
                elif 135 < angle <= 225: mask[1] = 1
                elif 225 < angle < 315: mask[3] = 1
            return mask
        return np.ones(self.num_blowers + self.num_dumps, dtype=np.int8)

    def _calculate_rewards(self):
        rew = {a: -0.01 for a in self.possible_agents}
        for i, b in enumerate(self.sim.blowers):
            cleared = getattr(b, 'snow_cleared_this_step', 0)
            if cleared > 0 and b.assigned_truck and b.assigned_truck.current_node == b.current_node:
                rew[f"blower_{i}"] += (5.0 * cleared)
                rew[f"truck_{self.sim.trucks.index(b.assigned_truck)}"] += (5.0 * cleared)
        return rew

    def _apply_blower_action(self, agent_id, action):
        idx = int(agent_id.split("_")[1]); blower = self.sim.blowers[idx]
        if action == 4: blower.route = []; return
        neighbors = list(self.sim.graph.successors(blower.current_node))
        if not neighbors: return
        cx, cy = float(self.sim.graph.nodes[blower.current_node]['x']), float(self.sim.graph.nodes[blower.current_node]['y'])
        targets = {0: 0, 1: 180, 2: 90, 3: 270}
        best_node, best_align = None, -float('inf')
        for n in neighbors:
            nx, ny = float(self.sim.graph.nodes[n]['x']), float(self.sim.graph.nodes[n]['y'])
            angle = (math.degrees(math.atan2(nx - cx, ny - cy)) + 360) % 360
            align = math.cos(math.radians(min(abs(targets[action] - angle), 360 - abs(targets[action] - angle))))
            if align > best_align: best_align, best_node = align, n
        if best_align > 0.5: blower.route = [best_node]

    def _apply_truck_action(self, agent_id, action):
        idx = int(agent_id.split("_")[1]); truck = self.sim.trucks[idx]
        if action < self.num_blowers:
            truck.blower, truck.target_dump = self.sim.blowers[action], None
        else:
            truck.blower, truck.target_dump = None, self.sim.dumps[action - self.num_blowers]

    def render(self):
        if not hasattr(self, 'fig'):
            plt.ion(); self.fig, self.ax = plt.subplots(figsize=(8,6))
            self.pos = {n: (float(d['x']), float(d['y'])) for n, d in self.sim.graph.nodes(data=True)}
        self.ax.clear()
        ec = ['#f5f6fa' if d.get('snow_depth', 0) > 0 else '#353b48' for u, v, k, d in self.sim.graph.edges(data=True, keys=True)]
        nx.draw_networkx_edges(self.sim.graph, self.pos, ax=self.ax, edge_color=ec, width=1, alpha=0.3)
        bx, by = zip(*[self.pos[b.current_node] for b in self.sim.blowers])
        self.ax.scatter(bx, by, c='blue', s=30); plt.draw(); plt.pause(0.001)
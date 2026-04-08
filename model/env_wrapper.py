import functools
import math
import random
import time
import numpy as np
import networkx as nx
from pettingzoo import ParallelEnv
from gymnasium import spaces
import pygame
from model.sim_engine import SnowRemovalSim, SnowBlower, DumpTruck 

class MontrealSnowEnv(ParallelEnv):
    metadata = {"name": "montreal_snow_v1"}

    def __init__(self, graph_filepath="data/plateau_mont_royal_drive.graphml", num_blowers=2, num_trucks=6, render_mode=None):
        self.graph_filepath = graph_filepath
        self.num_blowers = num_blowers
        self.num_trucks = num_trucks
        self.num_dumps = 2
        self.render_mode = render_mode
        self.possible_agents = (
            [f"blower_{i}" for i in range(num_blowers)] +
            [f"truck_{i}" for i in range(num_trucks)]
        )
        self.agents = []
        self.sim = None 
        
        # --- THE FIX: Load Graph & Calculate Dimensions ONCE ---
        print(f"Loading base environment graph from {graph_filepath}...")
        self.base_graph = nx.read_graphml(graph_filepath)
        
        for node, data in self.base_graph.nodes(data=True):
            data['x'] = float(data['x'])
            data['y'] = float(data['y'])
            
        self.num_nodes = len(self.base_graph.nodes())
        self.num_edges = len(self.base_graph.edges())
        
        # Precompute angles on the base graph to save CPU during training
        for u, v, k, data in self.base_graph.edges(data=True, keys=True):
            ux, uy = float(self.base_graph.nodes[u]['x']), float(self.base_graph.nodes[u]['y'])
            vx, vy = float(self.base_graph.nodes[v]['x']), float(self.base_graph.nodes[v]['y'])
            data['angle'] = (math.degrees(math.atan2(vx - ux, vy - uy)) + 360) % 360

    def get_task(self):
        return getattr(self, "curriculum_level", 1)

    def set_task(self, task):
        self.curriculum_level = task
        print(f"--- ENVIRONMENT UPGRADED TO CURRICULUM LEVEL {task} ---")
        

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        num_act = 5 if "blower" in agent else self.num_blowers + self.num_dumps
        return spaces.Discrete(num_act)

    def reset(self, seed=None, options=None):
        self.num_steps = 0
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            for a in self.possible_agents:
                self.action_space(a).seed(seed)
                self.observation_space(a).seed(seed)
                
        self.agents = self.possible_agents[:]
        
        # Pass the pre-loaded base_graph to the sim engine instead of the filepath
        self.sim = SnowRemovalSim(self.base_graph)
        
        self.node_to_idx = {n: i for i, n in enumerate(self.sim.nodes)}
        self.blower_to_idx = {b: i for i, b in enumerate(self.sim.blowers)}
        self._spawn_fleet()
        
        return self._get_graph_state(), {a: {} for a in self.possible_agents}

    def _spawn_fleet(self):
        # 1. Spawn Blowers (Always random)
        for i in range(self.num_blowers):
            start_node = random.choice(self.sim.nodes)
            self.sim.blowers.append(SnowBlower(self.sim.env, self.sim, f"blower_{i}", start_node))
            
        # 2. Spawn Trucks (Based on Curriculum Level)
        current_level = getattr(self, "curriculum_level", 1)
        
        for i in range(self.num_trucks):
            assigned_blower_node = self.sim.blowers[i % self.num_blowers].current_node
            
            if current_level == 1:
                # LEVEL 1: Spawn exactly on the Blower
                start_node = assigned_blower_node
                
            elif current_level == 2:
                # LEVEL 2: Spawn within a 2-block radius
                # nx.single_source_shortest_path_length gets all nodes within 'cutoff' distance
                neighbors = list(nx.single_source_shortest_path_length(self.sim.graph, assigned_blower_node, cutoff=2).keys())
                start_node = random.choice(neighbors)
                
            else:
                # LEVEL 3: Full City Random Spawn
                start_node = random.choice(self.sim.nodes)
                
            self.sim.trucks.append(DumpTruck(self.sim.env, self.sim, f"truck_{i}", start_node))

    def step(self, actions):
        # --- THE FIX: Reset the tracker variables for the new step ---
        for b in self.sim.blowers:
            b.snow_cleared_this_step = 0.0
        for t in self.sim.trucks:
            t.dumped_snow_this_step = False

        # Apply actions
        for agent_id, action in actions.items():
            if agent_id in self.agents:
                if "blower" in agent_id: self._apply_blower_action(agent_id, int(action))
                else: self._apply_truck_action(agent_id, int(action))
                
        # Advance the simulation
        self.sim.env.run(until=self.sim.env.now + 10)
        
        observations = self._get_graph_state()
        rewards = self._calculate_rewards()
        
        all_snow_cleared = self.sim.total_snow <= 0.1
        time_up = self.sim.env.now >  (60 * 60 * 4) # 4 hours in seconds
        is_done = all_snow_cleared or time_up
        
        terminations = {a: is_done for a in self.possible_agents}
        truncations = {a: is_done for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}

        if is_done:
            snow_removed = getattr(self.sim, 'initial_snow', 1000) - self.sim.total_snow
            time_elapsed_mins = self.sim.env.now / 60.0
            
            current_level = getattr(self, "curriculum_level", 1)
            for a in self.possible_agents:
                infos[a]["snow_removed"] = snow_removed
                infos[a]["time_elapsed_mins"] = time_elapsed_mins
                infos[a]["curriculum_level"] = current_level

            print(f"Episode ended. Snow removed: {snow_removed:.1f}. Time: {time_elapsed_mins:.1f} mins.")
            self.agents = []
            
        self.num_steps += 1
        return observations, rewards, terminations, truncations, infos

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        graph_obs_space = spaces.Dict({
            # --- NO MORE MAGIC NUMBERS ---
            "intersections": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_nodes, 2), dtype=np.float32), 
            "blowers": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_blowers, 2), dtype=np.float32),
            "trucks": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_trucks, 3), dtype=np.float32),
            "edges": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_edges, 1), dtype=np.float32),
            "agent_index": spaces.Box(low=0, high=max(self.num_blowers, self.num_trucks), shape=(1,), dtype=np.float32)
        })
        num_act = 5 if "blower" in agent else self.num_blowers + self.num_dumps
        return spaces.Dict({
            "observation": graph_obs_space,
            "action_mask": spaces.Box(low=0, high=1, shape=(num_act,), dtype=np.float32)
        })

    def _get_graph_state(self):
        # --- REMOVED MAGIC NUMBERS: Array shape is now dynamic (num_nodes, 2) ---
        int_feat = np.zeros((self.num_nodes, 2), dtype=np.float32)
        for i, node in enumerate(self.sim.nodes[:self.num_nodes]):
            # Feature 0: Traffic Flag
            if self.sim.intersections[node].count > 0: 
                int_feat[i, 0] = 1.0
                
            # Feature 1: The Hack! Sum up all the snow on the streets touching this intersection
            local_snow = 0.0
            # .edges(node) gets all streets connected to this specific node
            for u, v, k, d in self.sim.graph.edges(node, keys=True, data=True):
                local_snow += float(d.get('snow_depth', 0.0))
            int_feat[i, 1] = local_snow

        # Blowers
        b_feat = np.zeros((self.num_blowers, 2), dtype=np.float32)
        for i, b in enumerate(self.sim.blowers):
            b_feat[i, 0] = float(self.node_to_idx.get(b.current_node, -1))
            b_feat[i, 1] = 1.0 if getattr(b, 'is_waiting', False) else 0.0

        # Trucks
        t_feat = np.zeros((self.num_trucks, 3), dtype=np.float32)
        for i, t in enumerate(self.sim.trucks):
            t_feat[i, 0] = float(self.node_to_idx.get(t.current_node, -1))
            t_feat[i, 1] = float(t.payload / t.max_capacity)
            t_feat[i, 2] = float(self.blower_to_idx.get(t.blower, -1)) if t.blower else -1.0

        # --- REMOVED MAGIC NUMBERS: Array shape is now dynamic (num_edges, 1) ---
        e_feat = np.zeros((self.num_edges, 1), dtype=np.float32)
        for i, (u, v, k, d) in enumerate(self.sim.graph.edges(keys=True, data=True)):
            if i >= self.num_edges: break
            e_feat[i, 0] = float(d.get('snow_depth', 0.0))

        base = {"intersections": int_feat, "blowers": b_feat, "trucks": t_feat, "edges": e_feat}
        
        obs_dict = {}
        for a in self.possible_agents:
            # 1. Parse the integer ID from the agent string ("blower_0" -> 0)
            agent_idx = float(a.split("_")[1])
            
            # 2. Shallow copy the base map and inject the specific ID
            agent_obs = base.copy()
            agent_obs["agent_index"] = np.array([agent_idx], dtype=np.float32)
            
            obs_dict[a] = {
                "observation": agent_obs,
                "action_mask": np.array(self._get_mask(a), dtype=np.float32)
            }
            
        return obs_dict
    
    
    def _get_mask(self, agent_id):
        if "blower" in agent_id:
            idx = int(agent_id.split("_")[1]); blower = self.sim.blowers[idx]
            mask = np.zeros(5, dtype=np.int8)
            mask[4] = 1 # Action 4 (Stop/Wait) is always allowed
            
            # --- THE FIX: IN-TRANSIT NO-OP ---
            if getattr(blower, 'is_moving', False):
                return mask # Force network to Wait while driving
            
            # Use precomputed angles for O(1) lookups
            for n in self.sim.graph.successors(blower.current_node):
                angle = self.sim.graph[blower.current_node][n][0].get('angle', 0)
                if 315 <= angle or angle <= 45: mask[0] = 1
                elif 45 < angle <= 135: mask[2] = 1
                elif 135 < angle <= 225: mask[1] = 1
                elif 225 < angle < 315: mask[3] = 1
            return mask
            
        return np.ones(self.num_blowers + self.num_dumps, dtype=np.int8)

    def _calculate_rewards(self):
        rew = {a: 0.0 for a in self.possible_agents}
        
        # 1. Global Time Penalty (Encourages speed)
        for a in self.possible_agents:
            rew[a] -= 0.05 

        # 2. Blower Logic
        for i, b in enumerate(self.sim.blowers):
            b_id = f"blower_{i}"
            cleared = getattr(b, 'snow_cleared_this_step', 0)
            
            if cleared > 0:
                # --- UPDATED: Reward logic now allows for trailing node ---
                if b.assigned_truck and b.assigned_truck.current_node in (b.current_node, getattr(b, 'prev_node', b.current_node)):
                    # THE JACKPOT: Massive reward for successful collaboration
                    collaboration_reward = 15.0 * float(cleared)
                    rew[b_id] += collaboration_reward
                    t_idx = self.sim.trucks.index(b.assigned_truck)
                    rew[f"truck_{t_idx}"] += collaboration_reward
                else:
                    rew[b_id] += 0.1 
            elif getattr(b, 'is_waiting', False):
                rew[b_id] -= 0.2 

        # 3. Truck Logic
        for i, t in enumerate(self.sim.trucks):
            t_id = f"truck_{i}"
            
            # Distance-based breadcrumb trail
            if t.blower and t.payload < t.max_capacity:
                tx = float(self.sim.graph.nodes[t.current_node]['x'])
                ty = float(self.sim.graph.nodes[t.current_node]['y'])
                bx = float(self.sim.graph.nodes[t.blower.current_node]['x'])
                by = float(self.sim.graph.nodes[t.blower.current_node]['y'])
                dist = math.hypot(bx - tx, by - ty)
                rew[t_id] -= (dist * 0.001) 

            if getattr(t, 'dumped_snow_this_step', False):
                rew[t_id] += 200.0 
            
            if t.payload >= t.max_capacity:
                rew[t_id] -= 0.2

        # 4. Win Condition (Shared by all)
        if self.sim.total_snow <= 0.1:
            for a in self.possible_agents:
                rew[a] += 500.0
                
        return rew

    def _apply_blower_action(self, agent_id, action):
        idx = int(agent_id.split("_")[1]); blower = self.sim.blowers[idx]
        if action == 4: blower.route = []; return
        neighbors = list(self.sim.graph.successors(blower.current_node))
        if not neighbors: return
        
        # Use precomputed angles here too
        targets = {0: 0, 1: 180, 2: 90, 3: 270}
        best_node, best_align = None, -float('inf')
        for n in neighbors:
            angle = self.sim.graph[blower.current_node][n][0].get('angle', 0)
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
        """Draws the Graph, Dumps, Blowers, and Trucks using Pygame."""
        if self.render_mode != "rgb_array":
            return None

        if not hasattr(self, "screen"):
            pygame.init()
            self.window_size = 800
            self.screen = pygame.Surface((self.window_size, self.window_size))
            xs = [data['x'] for node, data in self.sim.graph.nodes(data=True)]
            ys = [data['y'] for node, data in self.sim.graph.nodes(data=True)]
            self.min_x, self.max_x = min(xs), max(xs)
            self.min_y, self.max_y = min(ys), max(ys)

        self.screen.fill((255, 255, 255)) 

        def scale_pos(x, y):
            norm_x = (x - self.min_x) / (self.max_x - self.min_x + 1e-6)
            norm_y = (y - self.min_y) / (self.max_y - self.min_y + 1e-6)
            padding = 40
            screen_x = padding + norm_x * (self.window_size - 2 * padding)
            screen_y = self.window_size - (padding + norm_y * (self.window_size - 2 * padding)) 
            return int(screen_x), int(screen_y)

        for u, v, k, data in self.sim.graph.edges(data=True, keys=True):
            ux, uy = self.sim.graph.nodes[u]['x'], self.sim.graph.nodes[u]['y']
            vx, vy = self.sim.graph.nodes[v]['x'], self.sim.graph.nodes[v]['y']
            snow = data.get('snow_depth', 0)
            if snow > 0:
                color, width = (200, 220, 255), 3
            else:
                color, width = (200, 200, 200), 1
            pygame.draw.line(self.screen, color, scale_pos(ux, uy), scale_pos(vx, vy), width)

        for dump_node in self.sim.dumps:
            dx, dy = self.sim.graph.nodes[dump_node]['x'], self.sim.graph.nodes[dump_node]['y']
            pos = scale_pos(dx, dy)
            pygame.draw.rect(self.screen, (0, 200, 0), (pos[0]-10, pos[1]-10, 20, 20))

        for b in self.sim.blowers:
            bx, by = self.sim.graph.nodes[b.current_node]['x'], self.sim.graph.nodes[b.current_node]['y']
            pygame.draw.circle(self.screen, (255, 140, 0), scale_pos(bx, by), 8)

        for t in self.sim.trucks:
            tx, ty = self.sim.graph.nodes[t.current_node]['x'], self.sim.graph.nodes[t.current_node]['y']
            pygame.draw.circle(self.screen, (0, 0, 255), scale_pos(tx, ty), 6)

        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2)) 
        return frame
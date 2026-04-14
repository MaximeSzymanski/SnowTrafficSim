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
    """
    A multi-agent reinforcement learning environment built on PettingZoo.
    Simulates a collaborative snow removal operation in Montreal using a
    graph-based road network. Agents consist of SnowBlowers (which clear snow)
    and DumpTrucks (which collect snow and transport it to depots).
    """
    metadata = {"name": "montreal_snow_v1"}

    def __init__(self, graph_filepath="data/plateau_mont_royal_drive.graphml", num_blowers=2, num_trucks=6, render_mode=None):
        """
        Initializes the environment, loads the map topology, and precomputes
        routing dictionaries to optimize simulation speed.
        """
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
        
        print(f"Loading base environment graph from {graph_filepath}...")
        self.base_graph = nx.MultiDiGraph(nx.read_graphml(graph_filepath))
        
        for node, data in self.base_graph.nodes(data=True):
            data['x'] = float(data['x'])
            data['y'] = float(data['y'])
            
        reverse_edges = []
        for u, v, k, data in self.base_graph.edges(data=True, keys=True):
            if not self.base_graph.has_edge(v, u):
                reverse_edges.append((v, u, data.copy()))
                
        for u, v, data in reverse_edges:
            self.base_graph.add_edge(u, v, **data)
            
        self.num_nodes = len(self.base_graph.nodes())
        self.num_edges = len(self.base_graph.edges())
        
        for u, v, k, data in self.base_graph.edges(data=True, keys=True):
            ux, uy = float(self.base_graph.nodes[u]['x']), float(self.base_graph.nodes[u]['y'])
            vx, vy = float(self.base_graph.nodes[v]['x']), float(self.base_graph.nodes[v]['y'])
            data['angle'] = (math.degrees(math.atan2(vx - ux, vy - uy)) + 360) % 360

        self.node_coords = {n: (data['x'], data['y']) for n, data in self.base_graph.nodes(data=True)}
        self.node_components = {}
        for i, comp in enumerate(nx.weakly_connected_components(self.base_graph)):
            for node in comp:
                self.node_components[node] = i

        self.ordered_edges = list(self.base_graph.edges(keys=True, data=True))

    def get_task(self):
        """
        Returns the current curriculum level of the environment.
        """
        return getattr(self, "curriculum_level", 1)

    def set_task(self, task):
        """
        Updates the curriculum level to increase task difficulty during training.
        """
        self.curriculum_level = task
        print(f"--- ENVIRONMENT UPGRADED TO CURRICULUM LEVEL {task} ---")
        
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Defines the discrete action space for a given agent. Blowers utilize 
        directional steering, while trucks utilize assignment selection.
        """
        num_act = 5 if "blower" in agent else self.num_blowers + self.num_dumps + 1
        return spaces.Discrete(num_act)

    def reset(self, seed=None, options=None):
        """
        Resets the simulation engine, drops a new snowstorm on the graph, 
        and respawns the fleet at their starting locations.
        """
        self.num_steps = 0
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            for a in self.possible_agents:
                self.action_space(a).seed(seed)
                self.observation_space(a).seed(seed)
                
        self.agents = self.possible_agents[:]
        self.sim = SnowRemovalSim(self.base_graph)
        
        self.node_to_idx = {n: i for i, n in enumerate(self.sim.nodes)}
        self.blower_to_idx = {b: i for i, b in enumerate(self.sim.blowers)}
        self._spawn_fleet()
        
        self.active_edge_data = [self.sim.graph[u][v][k] for u, v, k, d in self.ordered_edges]
        
        self.node_to_outgoing_edges = {n: [] for n in self.sim.nodes}
        for i, (u, v, k, d) in enumerate(self.ordered_edges):
            self.node_to_outgoing_edges[u].append(self.active_edge_data[i])
        
        return self._get_graph_state(), {a: {} for a in self.possible_agents}

    def _spawn_fleet(self):
        """
        Spawns snowblowers and dump trucks onto the physical graph based on 
        the current curriculum difficulty setting.
        """
        for i in range(self.num_blowers):
            start_node = random.choice(self.sim.nodes)
            self.sim.blowers.append(SnowBlower(self.sim.env, self.sim, f"blower_{i}", start_node))
            
        current_level = getattr(self, "curriculum_level", 1)
        
        for i in range(self.num_trucks):
            assigned_blower_node = self.sim.blowers[i % self.num_blowers].current_node
            
            if current_level == 1:
                start_node = assigned_blower_node
            elif current_level == 2:
                neighbors = list(nx.single_source_shortest_path_length(self.sim.graph, assigned_blower_node, cutoff=2).keys())
                start_node = random.choice(neighbors)
            else:
                start_node = random.choice(self.sim.nodes)
                
            self.sim.trucks.append(DumpTruck(self.sim.env, self.sim, f"truck_{i}", start_node))

    def step(self, actions):
        """
        Advances the simulation by applying fleet actions, checking loop deadlocks,
        calculating rewards, and returning standard multi-agent RL step tuples.
        """
        for b in self.sim.blowers:
            b.snow_cleared_this_step = 0.0
        for t in self.sim.trucks:
            t.dumped_snow_this_step = False

        for agent_id, action in actions.items():
            if agent_id in self.agents:
                if "blower" in agent_id: self._apply_blower_action(agent_id, int(action))
                else: self._apply_truck_action(agent_id, int(action))
                
        self.sim.env.run(until=self.sim.env.now + 10)
        
        loop_triggered = False
        for b in self.sim.blowers:
            b.is_looping = False
            if len(b.node_history) == 8 and len(set(b.node_history)) <= 3:
                b.is_looping = True
                loop_triggered = True
        
        observations = self._get_graph_state()
        rewards = self._calculate_rewards()
        
        all_snow_cleared = self.sim.total_snow <= 0.1
        time_up = self.sim.env.now >  (60 * 60 * 4) 
        
        is_done = all_snow_cleared or time_up or loop_triggered
        
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
        """
        Defines the multi-modal dictionary observation space required for the 
        Graph Neural Network policy architecture.
        """
        graph_obs_space = spaces.Dict({
            "intersections": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_nodes, 2), dtype=np.float32), 
            "blowers": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_blowers, 2), dtype=np.float32),
            "trucks": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_trucks, 3), dtype=np.float32),
            "edges": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_edges, 1), dtype=np.float32),
            "agent_index": spaces.Box(low=0, high=max(self.num_blowers, self.num_trucks), shape=(1,), dtype=np.float32)
        })
        num_act = 5 if "blower" in agent else self.num_blowers + self.num_dumps + 1
        return spaces.Dict({
            "observation": graph_obs_space,
            "action_mask": spaces.Box(low=0, high=1, shape=(num_act,), dtype=np.float32)
        })

    def _get_graph_state(self):
        """
        Compiles the current global state of the city grid, including snow load, 
        depot beacons, and specific vehicle telemetry into matrix features.
        """
        int_feat = np.zeros((self.num_nodes, 2), dtype=np.float32)
        for i, node in enumerate(self.sim.nodes[:self.num_nodes]):
            if node in self.sim.dumps:
                int_feat[i, 0] = 1.0  
            else:
                int_feat[i, 0] = 0.0
                
            int_feat[i, 1] = sum(d.get('snow_depth', 0.0) for d in self.node_to_outgoing_edges[node])

        b_feat = np.zeros((self.num_blowers, 2), dtype=np.float32)
        for i, b in enumerate(self.sim.blowers):
            b_feat[i, 0] = float(self.node_to_idx.get(b.current_node, -1))
            b_feat[i, 1] = 1.0 if getattr(b, 'is_waiting', False) else 0.0

        t_feat = np.zeros((self.num_trucks, 3), dtype=np.float32)
        for i, t in enumerate(self.sim.trucks):
            t_feat[i, 0] = float(self.node_to_idx.get(t.current_node, -1))
            t_feat[i, 1] = float(t.payload / t.max_capacity)
            t_feat[i, 2] = float(self.blower_to_idx.get(t.blower, -1)) if t.blower else -1.0

        e_feat = np.zeros((self.num_edges, 1), dtype=np.float32)
        for i, d in enumerate(self.active_edge_data):
            if i >= self.num_edges: break
            e_feat[i, 0] = float(d.get('snow_depth', 0.0))

        base = {"intersections": int_feat, "blowers": b_feat, "trucks": t_feat, "edges": e_feat}
        
        obs_dict = {}
        for a in self.possible_agents:
            agent_idx = float(a.split("_")[1])
            agent_obs = base.copy()
            agent_obs["agent_index"] = np.array([agent_idx], dtype=np.float32)
            obs_dict[a] = {
                "observation": agent_obs,
                "action_mask": np.array(self._get_mask(a), dtype=np.float32)
            }
        return obs_dict
    
    def _get_mask(self, agent_id):
        """
        Generates a valid action mask to prevent agents from violating the laws
        of physics (e.g., navigating through structures) or breaking dispatch rules.
        """
        if "blower" in agent_id:
            idx = int(agent_id.split("_")[1]); blower = self.sim.blowers[idx]
            mask = np.zeros(5, dtype=np.int8)
            mask[4] = 1 
            
            if getattr(blower, 'is_moving', False):
                return mask 
            
            successors = list(self.sim.graph.successors(blower.current_node))
            
            if not successors: 
                successors = list(self.sim.graph.predecessors(blower.current_node))
                
            prev_node = getattr(blower, 'prev_node', None)
            
            for n in successors:
                if n == prev_node and len(successors) > 1:
                    continue 
                
                try:
                    angle = self.sim.graph[blower.current_node][n][0].get('angle', 0)
                except KeyError:
                    angle = self.sim.graph[n][blower.current_node][0].get('angle', 0)
                    angle = (angle + 180) % 360 

                if 315 <= angle or angle <= 45: mask[0] = 1
                elif 45 < angle <= 135: mask[2] = 1
                elif 135 < angle <= 225: mask[1] = 1
                elif 225 < angle < 315: mask[3] = 1
                
            if np.sum(mask[:4]) == 0 and len(successors) > 0:
                mask[:4] = 1
                
            return mask
            
        idx = int(agent_id.split("_")[1])
        truck = self.sim.trucks[idx]
        num_actions = self.num_blowers + self.num_dumps + 1
        mask = np.zeros(num_actions, dtype=np.int8)
        
        wait_action_idx = num_actions - 1
        mask[wait_action_idx] = 1 
        
        if getattr(truck, 'is_moving', False) or getattr(truck, 'dumped_snow_this_step', False):
            if truck.blower:
                mask[self.sim.blowers.index(truck.blower)] = 1
                mask[wait_action_idx] = 0
            elif truck.target_dump:
                mask[self.num_blowers + self.sim.dumps.index(truck.target_dump)] = 1
                mask[wait_action_idx] = 0
            return mask
            
        truck_comp = self.node_components.get(truck.current_node, -1)
            
        if truck.payload < truck.max_capacity:
            for i, b in enumerate(self.sim.blowers):
                if truck_comp == self.node_components.get(b.current_node, -2):
                    mask[i] = 1 
            
        if truck.payload > 0:
            for i, d in enumerate(self.sim.dumps):
                if truck_comp == self.node_components.get(d, -2):
                    mask[self.num_blowers + i] = 1 
            
        return mask

    def _calculate_rewards(self):
        """
        Computes shaping rewards for agents, balancing collaboration metrics,
        distance penalties, anti-loop restrictions, and active fuel costs.
        """
        rew = {a: 0.0 for a in self.possible_agents}
        
        for a in self.possible_agents:
            rew[a] -= 0.05 

        for i, b in enumerate(self.sim.blowers):
            b_id = f"blower_{i}"
            cleared = getattr(b, 'snow_cleared_this_step', 0)
            
            if getattr(b, 'is_looping', False):
                rew[b_id] -= 100.0
                print(f"⚠️ {b_id} caught looping! Episode truncated to save CPU.")
                
            elif cleared > 0:
                if b.assigned_truck and b.assigned_truck.current_node in (b.current_node, getattr(b, 'prev_node', b.current_node)):
                    collaboration_reward = 15.0 * float(cleared)
                    rew[b_id] += collaboration_reward
                    t_idx = self.sim.trucks.index(b.assigned_truck)
                    rew[f"truck_{t_idx}"] += collaboration_reward
                else:
                    rew[b_id] += 0.1 
            elif getattr(b, 'is_waiting', False):
                rew[b_id] -= 0.2 
            else:
                rew[b_id] -= 0.1 

        for i, t in enumerate(self.sim.trucks):
            t_id = f"truck_{i}"
            
            if t.blower and t.payload < t.max_capacity:
                tx, ty = self.node_coords[t.current_node]
                bx, by = self.node_coords[t.blower.current_node]
                dist = math.hypot(bx - tx, by - ty)
                rew[t_id] -= (dist * 0.001) 
                
            elif t.target_dump and t.payload > 0:
                tx, ty = self.node_coords[t.current_node]
                dx, dy = self.node_coords[t.target_dump]
                dist = math.hypot(dx - tx, dy - ty)
                rew[t_id] -= (dist * 0.001) 

            if t.payload > 0:
                fuel_burn = (t.payload / t.max_capacity) * 0.5 
                rew[t_id] -= fuel_burn

            if getattr(t, 'dumped_snow_this_step', False):
                rew[t_id] += 200.0 
        
        if self.sim.total_snow <= 0.1:
            for a in self.possible_agents:
                rew[a] += 500.0
                
        return rew

    def _apply_blower_action(self, agent_id, action):
        """
        Translates a discrete neural network action into physical route 
        coordinates based on grid alignment.
        """
        idx = int(agent_id.split("_")[1]); blower = self.sim.blowers[idx]
        
        if action == 4: 
            if not getattr(blower, 'is_moving', False):
                blower.route = [] 
            return
            
        neighbors = list(self.sim.graph.successors(blower.current_node))
        
        if not neighbors: 
            neighbors = list(self.sim.graph.predecessors(blower.current_node))
            
        if not neighbors: return
        
        targets = {0: 0, 1: 180, 2: 90, 3: 270}
        best_node, best_align = None, -float('inf')
        
        for n in neighbors:
            try:
                angle = self.sim.graph[blower.current_node][n][0].get('angle', 0)
            except KeyError:
                angle = self.sim.graph[n][blower.current_node][0].get('angle', 0)
                angle = (angle + 180) % 360

            align = math.cos(math.radians(min(abs(targets[action] - angle), 360 - abs(targets[action] - angle))))
            if align > best_align: 
                best_align, best_node = align, n
                
        if len(neighbors) == 1:
            blower.route = [neighbors[0]]
        elif best_node is not None and best_align >= -0.1: 
            blower.route = [best_node]
        elif len(neighbors) > 0:
            blower.route = [random.choice(neighbors)]

    def _apply_truck_action(self, agent_id, action):
        """
        Handles assignment logic for the dump trucks, managing state transitions
        between blower chasing and depot dumping, whilst preventing ghost links.
        """
        idx = int(agent_id.split("_")[1]); truck = self.sim.trucks[idx]
        
        if action == self.num_blowers + self.num_dumps:
            return
            
        if action < self.num_blowers:
            target_b = self.sim.blowers[action]
            
            if truck.blower and truck.blower != target_b:
                if truck.blower.assigned_truck == truck:
                    truck.blower.assigned_truck = None 
                    
            truck.blower, truck.target_dump = target_b, None
            target_b.assigned_truck = truck 
            
        else:
            if truck.blower and truck.blower.assigned_truck == truck:
                truck.blower.assigned_truck = None 
                
            truck.blower, truck.target_dump = None, self.sim.dumps[action - self.num_blowers]

    def render(self, render_mode=None):
        """
        Renders a top-down visualizer of the simulation environment using PyGame,
        providing an RGB array for post-training video evaluation.
        """
        if self.render_mode != "rgb_array": return None
        
        if not hasattr(self, "screen"):
            pygame.init()
            pygame.font.init() 
            self.font = pygame.font.SysFont('Arial', 16, bold=True)
            self.small_font = pygame.font.SysFont('Arial', 14)
            
            self.map_size = 800
            self.sidebar_width = 250
            self.window_width = self.map_size + self.sidebar_width
            self.window_height = self.map_size
            
            self.screen = pygame.Surface((self.window_width, self.window_height))
            
            xs = [data['x'] for node, data in self.sim.graph.nodes(data=True)]
            ys = [data['y'] for node, data in self.sim.graph.nodes(data=True)]
            self.min_x, self.max_x = min(xs), max(xs)
            self.min_y, self.max_y = min(ys), max(ys)

        self.screen.fill((255, 255, 255)) 
        pygame.draw.rect(self.screen, (30, 30, 30), (0, 0, self.sidebar_width, self.window_height)) 

        y_offset = 20
        title = self.font.render("FLEET TELEMETRY", True, (255, 255, 255))
        self.screen.blit(title, (20, y_offset))
        y_offset += 40

        for i, t in enumerate(self.sim.trucks):
            t_text = self.small_font.render(f"Dump Truck {i}", True, (200, 200, 200))
            self.screen.blit(t_text, (20, y_offset))
            
            bar_x = 20
            bar_y = y_offset + 20
            bar_width = 200
            bar_height = 18
            
            pygame.draw.rect(self.screen, (80, 80, 80), (bar_x, bar_y, bar_width, bar_height))
            
            fill_pct = min(1.0, max(0.0, t.payload / t.max_capacity))
            fill_width = int(bar_width * fill_pct)
            if fill_width > 0:
                pygame.draw.rect(self.screen, (50, 150, 255), (bar_x, bar_y, fill_width, bar_height))
            
            pct_text = self.small_font.render(f"{t.payload:.1f} / {t.max_capacity:.1f} m³", True, (255, 255, 255))
            self.screen.blit(pct_text, (bar_x + 5, bar_y + 1))
            
            y_offset += 60

        y_offset += 20
        b_title = self.font.render("BLOWER STATS", True, (255, 255, 255))
        self.screen.blit(b_title, (20, y_offset))
        y_offset += 40

        for i, b in enumerate(self.sim.blowers):
            cleared = getattr(b, 'total_snow_cleared_lifetime', 0)
            b_text = self.small_font.render(f"Blower {i}: {cleared:.1f} m³ cleared", True, (255, 140, 0))
            self.screen.blit(b_text, (20, y_offset))
            y_offset += 30

        def scale_pos(x, y):
            norm_x = (x - self.min_x) / (self.max_x - self.min_x + 1e-6)
            norm_y = (y - self.min_y) / (self.max_y - self.min_y + 1e-6)
            padding = 40
            screen_x = self.sidebar_width + padding + norm_x * (self.map_size - 2 * padding)
            screen_y = self.map_size - (padding + norm_y * (self.map_size - 2 * padding)) 
            return int(screen_x), int(screen_y)

        drawn_edges = set()
        
        for u, v, k, data in self.sim.graph.edges(data=True, keys=True):
            edge_id = tuple(sorted((u, v)))
            if edge_id in drawn_edges:
                continue
            drawn_edges.add(edge_id)
            
            ux, uy = self.node_coords[u]
            vx, vy = self.node_coords[v]
            
            snow_1 = data.get('snow_depth', 0)
            snow_2 = 0
            if self.sim.graph.has_edge(v, u):
                snow_2 = self.sim.graph[v][u][0].get('snow_depth', 0)
                
            total_street_snow = snow_1 + snow_2
            
            if total_street_snow <= 0.2:
                color = (220, 220, 220) 
                width = 1
            else:
                color = (0, 120, 255)   
                width = 3
                
            pygame.draw.line(self.screen, color, scale_pos(ux, uy), scale_pos(vx, vy), width)

        for dump_node in self.sim.dumps:
            dx, dy = self.node_coords[dump_node]
            pos = scale_pos(dx, dy)
            pygame.draw.rect(self.screen, (0, 200, 0), (pos[0]-10, pos[1]-10, 20, 20))

        for b in self.sim.blowers:
            bx, by = self.node_coords[b.current_node]
            pygame.draw.circle(self.screen, (255, 140, 0), scale_pos(bx, by), 8)

        for t in self.sim.trucks:
            tx, ty = self.node_coords[t.current_node]
            pygame.draw.circle(self.screen, (0, 0, 255), scale_pos(tx, ty), 6)

        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2)) 
        return frame
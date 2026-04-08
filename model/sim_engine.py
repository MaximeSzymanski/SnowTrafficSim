import simpy
import networkx as nx
import random
import math

class SnowRemovalSim:
    # Change the parameter from graph_filepath to base_graph
    def __init__(self, base_graph):
        self.env = simpy.Environment()
        
        # --- THE FIX: Copy from memory, not from disk! ---
        self.graph = base_graph.copy()
        
        self.nodes = list(self.graph.nodes())
        
        # You can remove the cast-to-float loop for nodes because 
        # it's already done (or can be done once in env_wrapper)
        
        self.intersections = {
            node: simpy.Resource(self.env, capacity=10) for node in self.nodes
        }
        
        self.blowers = []
        self.trucks = []
        
        self.dumps = random.sample(self.nodes, 2)
        self.total_snow = 0.0
        
        # --- THE SNOW STORM ---
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            if 'length' in data:
                data['length'] = float(data['length'])
            
            # Reset snow depth for the new episode
            if 'snow_depth' not in data or float(data.get('snow_depth', 0)) == 0.0:
                data['snow_depth'] = random.uniform(5.0, 15.0)
            else:
                data['snow_depth'] = float(data['snow_depth'])
                
            self.total_snow += data['snow_depth']
            
        self.initial_snow = self.total_snow
        
        print(f"Total snow dropped on city: {self.total_snow:.1f} cubic meters")
        

class Vehicle:
    def __init__(self, env, sim_engine, vehicle_id, start_node):
        self.env = env
        self.sim = sim_engine
        self.id = vehicle_id
        self.current_node = start_node
        self.is_moving = False # <--- ADDED FLAG
        
        self.current_node_req = self.sim.intersections[self.current_node].request()
        self.action = env.process(self.run())

    def run(self):
        pass

    def drive_to(self, target_node):
        if not self.sim.graph.has_edge(self.current_node, target_node):
            yield self.env.timeout(1)
            return

        edge_data = self.sim.graph[self.current_node][target_node][0]
        length_m = float(edge_data.get('length', 100))
        
        maxspeed = edge_data.get('maxspeed', '30')
        if isinstance(maxspeed, list): maxspeed = maxspeed[0]
        try: speed_kmh = float(maxspeed)
        except ValueError: speed_kmh = 30.0 
            
        travel_time_seconds = length_m / (speed_kmh / 3.6)
        
        # 1. Travel down the street
        self.is_moving = True # <--- LOCK ACTION MASK
        yield self.env.timeout(travel_time_seconds)
        self.is_moving = False # <--- UNLOCK ACTION MASK
        
        # 2. Request the lock for the upcoming intersection
        target_req = self.sim.intersections[target_node].request()
        yield target_req
        
        # 3. Release the old intersection
        if self.current_node_req is not None:
            self.sim.intersections[self.current_node].release(self.current_node_req)
        self.current_node_req = target_req
        
        self.current_node = target_node

class SnowBlower(Vehicle):
    def __init__(self, env, sim_engine, vehicle_id, start_node):
        super().__init__(env, sim_engine, vehicle_id, start_node)
        self.route = [] 
        self.assigned_truck = None
        self.is_waiting = False
        self.snow_cleared_this_step = 0
        self.prev_node = start_node # <--- ADDED FOR REWARD FUNCTION ACCESS

    def run(self):
        yield self.current_node_req
        
        while True:
            if not self.route:
                self.is_waiting = False
                yield self.env.timeout(1)
                continue

            next_node = self.route[0]

            if self.assigned_truck:
                t_node = self.assigned_truck.current_node
                is_with_us = (t_node == self.current_node)
                is_trailing = self.sim.graph.has_edge(t_node, self.current_node) or self.sim.graph.has_edge(self.current_node, t_node)
                
                if not (is_with_us or is_trailing):
                    self.is_waiting = True
                    yield self.env.timeout(1)
                    continue
            
            self.is_waiting = False

            # Save where we started (on class attribute for reward function)
            self.prev_node = self.current_node 
            
            # PERFORM THE DRIVE
            yield self.env.process(self.drive_to(next_node))

            if self.sim.graph.has_edge(self.prev_node, self.current_node):
                edge_data = self.sim.graph[self.prev_node][self.current_node][0]
                snow_on_edge = edge_data.get('snow_depth', 0.0)
                
                if snow_on_edge > 0:
                    valid_trucks = [
                        t for t in self.sim.trucks 
                        if t.current_node in (self.current_node, self.prev_node) and t.payload < t.max_capacity
                    ]
                    
                    if valid_trucks:
                        assigned_valid = [t for t in valid_trucks if t == self.assigned_truck]
                        active_truck = assigned_valid[0] if assigned_valid else valid_trucks[0]
                        
                        available_space = active_truck.max_capacity - active_truck.payload
                        cleared = min(snow_on_edge, available_space)
                        
                        edge_data['snow_depth'] -= cleared
                        active_truck.payload += cleared
                        self.snow_cleared_this_step += cleared
                        self.sim.total_snow -= cleared

            if len(self.route) > 0 and self.route[0] == next_node:
                self.route.pop(0)
            
            self.is_waiting = False

class DumpTruck(Vehicle):
    def __init__(self, env, sim_engine, vehicle_id, start_node):
        super().__init__(env, sim_engine, vehicle_id, start_node)
        self.blower = None
        self.target_dump = None
        self.max_capacity = 50.0 
        self.payload = 0.0
        self.dumped_snow_this_step = False
        self._cached_route = []
        self._last_target_node = None

    def run(self):
        yield self.current_node_req
        
        while True:
            if self.target_dump is not None:
                if self.current_node != self.target_dump:
                    if not self._cached_route or self._last_target_node != self.target_dump:
                        try:
                            self._cached_route = nx.shortest_path(self.sim.graph, self.current_node, self.target_dump, weight='length')
                            self._last_target_node = self.target_dump
                            self._cached_route.pop(0) 
                        except nx.NetworkXNoPath:
                            self._cached_route = []
                            
                    if self._cached_route:
                        next_node = self._cached_route.pop(0)
                        yield self.env.process(self.drive_to(next_node))
                    else:
                        yield self.env.timeout(1)
                        
                elif self.current_node in self.sim.dumps and self.payload > 0.0 and not self.dumped_snow_this_step:
                    yield self.env.timeout(30)
                    self.payload = 0.0
                    self.dumped_snow_this_step = True
                    self.target_dump = None
                    self._cached_route = [] 
                    self._last_target_node = None
                    print(f"[{self.env.now:.1f}s] {self.id} emptied payload at Dump.")
                else:
                    self.target_dump = None 
                    self._cached_route = [] 
                    self._last_target_node = None
                    yield self.env.timeout(1)

            elif self.blower is not None:
                target_node = self.blower.current_node
                if self.current_node != target_node:
                    if not self._cached_route or self._last_target_node != target_node:
                        try:
                            self._cached_route = nx.shortest_path(self.sim.graph, self.current_node, target_node, weight='length')
                            self._last_target_node = target_node
                            self._cached_route.pop(0)
                        except nx.NetworkXNoPath:
                            self._cached_route = []
                            
                    if self._cached_route:
                        next_node = self._cached_route.pop(0)
                        yield self.env.process(self.drive_to(next_node))
                    else:
                        yield self.env.timeout(1)
                else:
                    self._cached_route = [] 
                    self._last_target_node = None
                    yield self.env.timeout(1)
            else:
                self._cached_route = [] 
                self._last_target_node = None
                yield self.env.timeout(1)
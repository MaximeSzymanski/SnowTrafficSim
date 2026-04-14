import simpy
import networkx as nx
import random
import math

ROUTE_CACHE = {}

class SnowRemovalSim:
    """
    Core simulation engine managing the physical environment, road network,
    and snow distribution for the multi-agent Montreal snow removal environment.
    """
    def __init__(self, base_graph):
        """
        Initializes the SimPy environment, populates the road network with initial
        snow depths, and allocates intersection resource capacities.
        """
        self.env = simpy.Environment()
        self.graph = base_graph.copy()
        self.nodes = list(self.graph.nodes())
        
        self.intersections = {
            node: simpy.Resource(self.env, capacity=10) for node in self.nodes
        }
        
        self.blowers = []
        self.trucks = []
        
        self.dumps = random.sample(self.nodes, 2)
        self.total_snow = 0.0
        
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            if 'length' in data: 
                data['length'] = float(data['length'])
            
            if 'snow_depth' not in data or float(data.get('snow_depth', 0)) == 0.0:
                data['snow_depth'] = random.uniform(5.0, 15.0)
            else:
                data['snow_depth'] = float(data['snow_depth'])
                
            self.total_snow += data['snow_depth']
            
        self.initial_snow = self.total_snow
        print(f"Total snow dropped on city: {self.total_snow:.1f} cubic meters")
        

class Vehicle:
    """
    Base agent class handling physical movement, timing, and resource requests
    within the SimPy environment.
    """
    def __init__(self, env, sim_engine, vehicle_id, start_node):
        """
        Initializes the vehicle's state, requests the initial node resource,
        and triggers the SimPy event loop.
        """
        self.env = env
        self.sim = sim_engine
        self.id = vehicle_id
        self.current_node = start_node
        self.is_moving = False 
        
        self.current_node_req = self.sim.intersections[self.current_node].request()
        self.action = env.process(self.run())

    def run(self):
        """
        Abstract method defining the vehicle's core behavior loop.
        """
        pass

    def drive_to(self, target_node):
        """
        Simulates physical travel time between connected nodes based on
        edge length and speed limits.
        """
        if not self.sim.graph.has_edge(self.current_node, target_node):
            yield self.env.timeout(1)
            return

        edge_data = self.sim.graph[self.current_node][target_node][0]
        length_m = float(edge_data.get('length', 100))
        
        maxspeed = edge_data.get('maxspeed', '30')
        if isinstance(maxspeed, list): 
            maxspeed = maxspeed[0]
            
        try: 
            speed_kmh = float(maxspeed)
        except ValueError: 
            speed_kmh = 30.0 
            
        travel_time_seconds = length_m / (speed_kmh / 3.6)
        
        self.is_moving = True 
        yield self.env.timeout(travel_time_seconds)
        self.is_moving = False 
        
        target_req = self.sim.intersections[target_node].request()
        yield target_req
        
        if self.current_node_req is not None:
            self.sim.intersections[self.current_node].release(self.current_node_req)
        self.current_node_req = target_req
        
        self.current_node = target_node


class SnowBlower(Vehicle):
    """
    Active agent responsible for navigating the graph geometry and transferring
    snow from the street edges into an assigned DumpTruck.
    """
    def __init__(self, env, sim_engine, vehicle_id, start_node):
        """
        Initializes blower-specific tracking metrics and anti-loop memory structures.
        """
        super().__init__(env, sim_engine, vehicle_id, start_node)
        self.route = [] 
        self.assigned_truck = None
        self.is_waiting = False
        
        self.snow_cleared_this_step = 0
        self.total_snow_cleared_lifetime = 0.0 
        self.prev_node = start_node 
        self.node_history = [] 
        self.is_looping = False 

    def run(self):
        """
        Main simulation loop. Navigates assigned routes, halts if unaccompanied
        by a valid truck, and deducts snow depth from the physical road network.
        """
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
            self.prev_node = self.current_node 
            
            yield self.env.process(self.drive_to(next_node))

            if self.current_node != self.prev_node:
                self.node_history.append(self.current_node)
                if len(self.node_history) > 8:
                    self.node_history.pop(0)

            if self.sim.graph.has_edge(self.prev_node, self.current_node):
                edge = self.sim.graph[self.prev_node][self.current_node][0]
            elif self.sim.graph.has_edge(self.current_node, self.prev_node):
                edge = self.sim.graph[self.current_node][self.prev_node][0]
            else:
                edge = None

            if edge is not None:
                snow_on_edge = edge.get('snow_depth', 0.0)
                
                if snow_on_edge > 0:
                    valid_nodes = [self.current_node, self.prev_node] + list(self.sim.graph.successors(self.current_node))
                    
                    valid_trucks = [
                        t for t in self.sim.trucks 
                        if t.current_node in valid_nodes and t.payload < t.max_capacity
                    ]
                    
                    if valid_trucks:
                        assigned_valid = [t for t in valid_trucks if t == self.assigned_truck]
                        active_truck = assigned_valid[0] if assigned_valid else valid_trucks[0]
                        
                        available_space = active_truck.max_capacity - active_truck.payload
                        cleared = min(snow_on_edge, available_space)
                        
                        edge['snow_depth'] -= cleared
                        active_truck.payload += cleared
                        self.snow_cleared_this_step += cleared
                        self.total_snow_cleared_lifetime += cleared 
                        self.sim.total_snow -= cleared

            if len(self.route) > 0 and self.route[0] == next_node:
                self.route.pop(0)
            
            self.is_waiting = False
            

class DumpTruck(Vehicle):
    """
    Support agent responsible for catching snow from blowers, navigating the
    shortest path to depot locations, and emptying payloads.
    """
    def __init__(self, env, sim_engine, vehicle_id, start_node):
        """
        Initializes truck capacity bounds, payload metrics, and path caching variables.
        """
        super().__init__(env, sim_engine, vehicle_id, start_node)
        self.blower = None
        self.target_dump = None
        self.max_capacity = 50.0 
        self.payload = 0.0
        self.dumped_snow_this_step = False
        self._cached_route = []
        self._last_target_node = None

    def _get_route(self, target_node):
        """
        Retrieves or computes the shortest path to a destination node, leveraging
        a global cache to drastically reduce Dijkstra calculation overhead.
        """
        route_key = (self.current_node, target_node)
        
        if route_key not in ROUTE_CACHE:
            try:
                ROUTE_CACHE[route_key] = nx.shortest_path(self.sim.graph, self.current_node, target_node, weight='length')
            except nx.NetworkXNoPath:
                ROUTE_CACHE[route_key] = None 
                
        cached_path = ROUTE_CACHE[route_key]
        
        if cached_path is None:
            return [] 
            
        route_copy = cached_path[:]
        if len(route_copy) > 0:
            route_copy.pop(0) 
            
        return route_copy

    def run(self):
        """
        Main simulation loop. Manages complex state transitions between chasing a
        target blower, navigating to a dump node, and undergoing dumping protocols.
        """
        yield self.current_node_req
        
        while True:
            if self.target_dump is not None:
                if self.current_node != self.target_dump:
                    if not self._cached_route or self._last_target_node != self.target_dump:
                        
                        self._cached_route = self._get_route(self.target_dump)
                        self._last_target_node = self.target_dump
                        
                        if not self._cached_route:
                            self.target_dump = None
                            self.blower = None
                            
                    if self._cached_route:
                        next_node = self._cached_route.pop(0)
                        yield self.env.process(self.drive_to(next_node))
                    else:
                        yield self.env.timeout(1)
                        
                elif self.current_node in self.sim.dumps and self.payload > 0.0 and not self.dumped_snow_this_step:
                    self.is_moving = True
                    yield self.env.timeout(30)
                    self.is_moving = False
                    
                    self.payload = 0.0
                    self.dumped_snow_this_step = True
                    self.target_dump = None
                    self._cached_route = [] 
                    self._last_target_node = None
                    dump_number = self.sim.dumps.index(self.current_node) + 1
                    print(f"[{self.env.now:.1f}s] {self.id} emptied payload at Dump number {dump_number} ({self.current_node})")
                else:
                    self.target_dump = None 
                    self._cached_route = [] 
                    self._last_target_node = None
                    yield self.env.timeout(1)

            elif self.blower is not None:
                target_node = self.blower.current_node
                if self.current_node != target_node:
                    if not self._cached_route or self._last_target_node != target_node:
                        
                        self._cached_route = self._get_route(target_node)
                        self._last_target_node = target_node
                        
                        if not self._cached_route:
                            self.target_dump = None
                            self.blower = None
                            
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
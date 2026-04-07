import simpy
import networkx as nx
import random
import math

class SnowRemovalSim:
    def __init__(self, graph_filepath):
        self.env = simpy.Environment()
        
        print("Loading Montreal road network...")
        self.graph = nx.read_graphml(graph_filepath)
        
        self.nodes = list(self.graph.nodes())
        print(f"Graph loaded with {len(self.nodes)} nodes and {len(self.graph.edges())} edges.")
        # Cast data nodes coordinates to floats for later math (OSMnx sometimes saves them as strings)
        for node, data in self.graph.nodes(data=True):
            data['x'] = float(data['x'])
            data['y'] = float(data['y'])
            
        # Case edge attributes (like 'length' and 'maxspeed') to floats/ints for physics calculations
        
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            # Length is critical for travel time and volume
            if 'length' in data:
                data['length'] = float(data['length'])
            
            # Snow depth was injected by our script, but ensure it's a float
            if 'snow_depth' in data:
                data['snow_depth'] = float(data['snow_depth'])
            else:
                data['snow_depth'] = 0.0
                
                
        # Every intersection is a strict lock (capacity=1)
        self.intersections = {
            node: simpy.Resource(self.env, capacity=1) for node in self.nodes
        }
        
        self.blowers = []
        self.trucks = []
        
        # Select 2 random nodes on the edge of the map to act as Snow Dumps
        self.dumps = random.sample(self.nodes, 2)
        print(f"Snow Dumps established at nodes: {self.dumps}")

class Vehicle:
    def __init__(self, env, sim_engine, vehicle_id, start_node):
        self.env = env
        self.sim = sim_engine
        self.id = vehicle_id
        self.current_node = start_node
        
        # Claim the starting lock immediately
        self.current_node_req = self.sim.intersections[self.current_node].request()
        self.action = env.process(self.run())

    def run(self):
        pass

    def drive_to(self, target_node):
        """Calculates travel time, requests the next lock, and releases the old lock."""
        if not self.sim.graph.has_edge(self.current_node, target_node):
            # Fallback if A* or NN makes a weird request
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
        yield self.env.timeout(travel_time_seconds)
        
        # 2. Request the lock for the upcoming intersection (Queue here if busy)
        target_req = self.sim.intersections[target_node].request()
        yield target_req
        
        # 3. Release the old intersection so traffic behind us can flow
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

    def run(self):
        yield self.current_node_req
        
        while True:
            # 1. Wait for an assignment
            if not self.route:
                self.is_waiting = False
                yield self.env.timeout(1)
                continue

            # 2. Identify target (but don't pop yet!)
            next_node = self.route[0]

            # 3. Master-Slave Sync: Wait for truck
            if self.assigned_truck:
                if self.assigned_truck.current_node != self.current_node:
                    self.is_waiting = True
                    yield self.env.timeout(1)
                    continue
            self.is_waiting = False

            # 4. PERFORM THE DRIVE
            # During this 'yield', the 'env.step()' function in your test script 
            # might run and change 'self.route'.
            yield self.env.process(self.drive_to(next_node))

            # 5. POST-DRIVE: Snow Clearing Math
            # We check if the edge still exists (just in case the graph changed)
            if self.sim.graph.has_edge(self.current_node, next_node):
                # We use target_node because self.current_node is now the arrival node
                # Note: drive_to updates self.current_node at the very end
                # So here, self.current_node IS next_node. We need the edge behind us.
                # Let's find the edge we just traversed.
                prev_node = None
                for u, v in self.sim.graph.edges(): # This is slow, better way:
                    pass 
                
                # Simplified: use the next_node we stored before the yield
                # Note: In drive_to, self.current_node was updated to next_node.
                # To get the edge data, we need the node we came FROM.
                # Let's assume the clearing happens *during* drive_to or we store the source.
                
                # For now, let's just fix the crash:
                pass

            # --- THE CRITICAL SAFETY CHECK ---
            # If the route was cleared or changed while we were driving, 
            # do NOT try to pop.
            if len(self.route) > 0 and self.route[0] == next_node:
                self.route.pop(0)
            
            # If the route is now empty, we are done with this macro-action
            self.is_waiting = False

class DumpTruck(Vehicle):
    def __init__(self, env, sim_engine, vehicle_id, start_node):
        super().__init__(env, sim_engine, vehicle_id, start_node)
        # PettingZoo Attributes
        self.blower = None
        self.target_dump = None
        
        # Physics Attributes
        self.max_capacity = 50.0 # cubic meters of snow
        self.payload = 0.0
        self.dumped_snow_this_step = False

    def run(self):
        yield self.current_node_req
        
        while True:
            # OPTION A: Route to Snow Dump
            if self.target_dump is not None:
                if self.current_node != self.target_dump:
                    try:
                        # Find the shortest path to the dump
                        path = nx.shortest_path(self.sim.graph, self.current_node, self.target_dump, weight='length')
                        if len(path) > 1:
                            yield self.env.process(self.drive_to(path[1]))
                        else:
                            yield self.env.timeout(1)
                    except nx.NetworkXNoPath:
                        yield self.env.timeout(1)
                elif self.current_node in self.sim.dumps and self.payload > 0.0 and not self.dumped_snow_this_step:
                    # We arrived at the dump! Unload snow.
                    yield self.env.timeout(30) # Takes 60 seconds to dump
                    self.payload = 0.0
                    self.dumped_snow_this_step = True
                    self.target_dump = None # Action complete
                    print(f"[{self.env.now:.1f}s] {self.id} emptied payload at Dump.")

            # OPTION B: Follow Assigned Master Blower
            elif self.blower is not None:
                if self.current_node != self.blower.current_node:
                    try:
                        # Find shortest path to catch up to the Master
                        path = nx.shortest_path(self.sim.graph, self.current_node, self.blower.current_node, weight='length')
                        if len(path) > 1:
                            yield self.env.process(self.drive_to(path[1]))
                        else:
                            yield self.env.timeout(1)
                    except nx.NetworkXNoPath:
                        yield self.env.timeout(1)
                else:
                    # Successfully synced with Blower! Idle and catch snow.
                    yield self.env.timeout(1)

            # OPTION C: Idle
            else:
                yield self.env.timeout(1)
import simpy
import networkx as nx
import random
import math

class SnowRemovalSim:
    def __init__(self, graph_filepath):
        # 1. Initialize the Simulation Environment
        self.env = simpy.Environment()
        
        # 2. Load the Montreal Graph
        print("Loading Montreal road network...")
        self.graph = nx.read_graphml(graph_filepath)
        
        # We need a list of valid intersection IDs to spawn our vehicles
        self.nodes = list(self.graph.nodes())
        
        self.intersections = {
            node: simpy.Resource(self.env, capacity=1) for node in self.nodes
        }
        # 3. Fleet tracking
        self.blowers = []
        self.trucks = []

class Vehicle:
    def __init__(self, env, sim_engine, vehicle_id, start_node):
        self.env = env
        self.sim = sim_engine
        self.vehicle_id = vehicle_id
        self.current_node = start_node
        
        self.current_node_req = self.sim.intersections[self.current_node].request()
        # Start the vehicle's lifecycle process
        self.action = env.process(self.run())

    def run(self):
        # This will be overridden by specific vehicle types
        pass

    def drive_to(self, target_node):
        """Calculates travel time and moves the vehicle along an edge."""
        # 1. Check if the edge exists in our Montreal graph
        if not self.sim.graph.has_edge(self.current_node, target_node):
            print(f"[{self.env.now}] Error: {self.vehicle_id} cannot drive from {self.current_node} to {target_node}. No direct street.")
            return

        # 2. Extract edge data (OSMnx edges are MultiDiGraphs, so we take the first key [0])
        edge_data = self.sim.graph[self.current_node][target_node][0]
        
        # Extract length (meters) and speed limit (km/h)
        length_m = float(edge_data.get('length', 100)) # Default to 100m if missing
        
        # OSMnx speed limits can be messy strings like "50" or "['30', '50']"
        maxspeed = edge_data.get('maxspeed', '30')
        if isinstance(maxspeed, list):
            maxspeed = maxspeed[0]
        try:
            speed_kmh = float(maxspeed)
        except ValueError:
            speed_kmh = 30.0 # Default fallback speed
            
        # 3. Calculate time: t = d / v
        speed_ms = speed_kmh / 3.6
        travel_time_seconds = length_m / speed_ms
        
        print(f"[{self.env.now:.1f}s] {self.vehicle_id} departing {self.current_node} towards {target_node}. ETA: {travel_time_seconds:.1f}s")
        
        # 4. SIMPY YIELD: This pauses the vehicle's code while simulation time passes
        yield self.env.timeout(travel_time_seconds)
        
        target_req = self.sim.intersections[target_node].request()
        yield target_req
        
        if self.current_node_req is not None:
            self.sim.intersections[self.current_node].release(self.current_node_req)
        self.current_node_req = target_req
        
        self.current_node = target_node
        print(f"[{self.env.now:.1f}s] {self.vehicle_id} arrived at {self.current_node}.")

class SnowBlower(Vehicle):
    def run(self):
        """For testing: A blower just wanders randomly clearing snow."""
        for _ in range(3): # Do 3 random moves
            neighbors = list(self.sim.graph.successors(self.current_node))
            if not neighbors:
                break
            
            # Pick a random connected street
            next_node = random.choice(neighbors)
            yield self.env.process(self.drive_to(next_node))

if __name__ == "__main__":
    # Ensure this points to the graph you saved!
    graph_file = "data/plateau_mont_royal_drive.graphml"
    
    sim = SnowRemovalSim(graph_file)
    
    # Spawn a Snow Blower at a random intersection
    start_loc = random.choice(sim.nodes)
    blower1 = SnowBlower(sim.env, sim, "Blower-01", start_loc)
    sim.blowers.append(blower1)
    
    print("--- Starting Simulation ---")
    # Run the simulation for exactly 1 hour (3600 seconds) of simulated time
    sim.env.run(until=3600)
    print("--- Simulation Ended ---")
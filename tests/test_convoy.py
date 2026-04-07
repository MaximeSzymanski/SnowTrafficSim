import simpy
import networkx as nx
import matplotlib.pyplot as plt

class TrafficSim:
    def __init__(self):
        self.env = simpy.Environment()
        # 5 intersections in a straight line
        self.graph = nx.path_graph(5)
        
        self.intersections = {
            node: simpy.Resource(self.env, capacity=1)
            for node in self.graph.nodes()
        }

class SnowBlower:
    def __init__(self, env, sim, b_id, start_node):
        self.env = env
        self.sim = sim
        self.id = b_id
        self.current_node = start_node
        self.assigned_truck = None
        
        self.req = sim.intersections[start_node].request()
        self.action = env.process(self.run())

    def run(self):
        yield self.req
        print(f"[{self.env.now:02d}s] {self.id} deployed at Node {self.current_node}")

        route = [1, 2, 3, 4] 
        
        for next_node in route:
            # Sync: Wait for truck
            if self.assigned_truck:
                while self.assigned_truck.current_node != self.current_node:
                    yield self.env.timeout(1)
            
            print(f"[{self.env.now:02d}s] Convoy synced! {self.id} clearing to Node {next_node}...")
            
            # Request next lock
            next_req = self.sim.intersections[next_node].request()
            yield next_req 
            
            # Clear the street (Takes 4 seconds)
            yield self.env.timeout(4)
            
            # Release old intersection
            self.sim.intersections[self.current_node].release(self.req)
            
            self.current_node = next_node
            self.req = next_req

class DumpTruck:
    def __init__(self, env, sim, t_id, start_node, assigned_blower):
        self.env = env
        self.sim = sim
        self.id = t_id
        self.current_node = start_node
        self.blower = assigned_blower
        self.blower.assigned_truck = self 
        
        self.req = sim.intersections[start_node].request()
        self.action = env.process(self.run())

    def run(self):
        yield self.req
        print(f"[{self.env.now:02d}s] {self.id} deployed at Node {self.current_node}")
        
        while True:
            # Follow Master
            if self.current_node != self.blower.current_node:
                target = self.blower.current_node
                
                target_req = self.sim.intersections[target].request()
                yield target_req 
                
                # Drive to master (Takes 3 seconds - faster than clearing)
                yield self.env.timeout(3) 
                
                self.sim.intersections[self.current_node].release(self.req)
                self.current_node = target
                self.req = target_req
                print(f"[{self.env.now:02d}s] {self.id} caught up to {self.blower.id} at Node {self.current_node}")
            
            yield self.env.timeout(1)

def visualizer(env, sim, blower, truck):
    """The Observer Process: Renders the graph every simulation second."""
    plt.ion() # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.canvas.manager.set_window_title('Convoy Logic Validator')
    
    # Position nodes in a straight horizontal line
    pos = {i: (i, 0) for i in range(len(sim.graph.nodes))}
    
    while True:
        ax.clear()
        
        # 1. Draw the street network
        nx.draw(sim.graph, pos, ax=ax, with_labels=True, 
                node_color='darkgrey', node_size=800, 
                edge_color='grey', width=3, font_color='white', font_weight='bold')
        
        # 2. Draw the Blower (Blue Circle) slightly above the intersection
        b_x, b_y = pos[blower.current_node]
        ax.plot(b_x, b_y + 0.15, marker='o', color='#00a8ff', markersize=20, label=f'{blower.id}')
        
        # 3. Draw the Truck (Red Square) slightly below the intersection
        t_x, t_y = pos[truck.current_node]
        ax.plot(t_x, t_y - 0.15, marker='s', color='#e84118', markersize=20, label=f'{truck.id}')
        
        # 4. Styling
        ax.set_title(f"Sim Time: {env.now:02d}s | Blue=Blower, Red=Truck", fontsize=14, pad=10)
        ax.set_ylim(-0.5, 0.5) 
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.draw()
        plt.pause(0.2) # Real-world pause so your eyes can track it
        
        yield env.timeout(1) # Sleep for 1 simulation second

if __name__ == "__main__":
    sim = TrafficSim()
    
    # Spawn agents
    blower1 = SnowBlower(sim.env, sim, "Blower-1", start_node=0)
    truck1 = DumpTruck(sim.env, sim, "Truck-1", start_node=0, assigned_blower=blower1)
    
    # Inject the visualizer process into the SimPy environment
    sim.env.process(visualizer(sim.env, sim, blower1, truck1))
    
    # Run the simulation
    sim.env.run(until=500)
    
    # Keep window open at the end
    plt.ioff()
    plt.show()
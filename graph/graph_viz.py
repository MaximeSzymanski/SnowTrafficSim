import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot_snow_gradient(filepath):
    print("Loading graph...")
    # Load the graph we saved in the previous step
    #G = nx.read_graphml(filepath)
    
    G = ox.load_graphml(filepath)

    # 1. Find the geographic boundaries to calculate the gradient
    y_coords = [float(data['y']) for node, data in G.nodes(data=True)]
    min_y, max_y = min(y_coords), max(y_coords)

    print("Calculating snow depths...")
    snow_depths = []
    
    # 2. Inject the 'snow_depth' feature into every edge
    # OSMnx uses MultiDiGraphs, so we iterate with keys
    for u, v, key, data in G.edges(keys=True, data=True):
        # Get the latitude of the nodes connecting this street
        y_u = float(G.nodes[u]['y'])
        y_v = float(G.nodes[v]['y'])
        avg_y = (y_u + y_v) / 2
        
        # Normalize the latitude to a 0.0 - 1.0 scale
        normalized_y = (avg_y - min_y) / (max_y - min_y)
        
        # Simulate a storm: 5cm minimum, up to 30cm at the highest latitude
        depth = 5 + (25 * normalized_y)
        
        # Save it to the graph edge features
        data['snow_depth'] = depth
        snow_depths.append(depth)

    # 3. Create the visual gradient (Colormap)
    print("Rendering visualization...")
    cmap = plt.cm.Blues # A nice icy blue fade
    norm = mcolors.Normalize(vmin=min(snow_depths), vmax=max(snow_depths))
    
    # Map each edge's snow depth to a specific color on the gradient
    edge_colors = [cmap(norm(float(data['snow_depth']))) for u, v, key, data in G.edges(keys=True, data=True)]

    # 4. Plot the graph
    fig, ax = ox.plot_graph(
        G,
        node_size=0,           # Hide nodes to make the street colors pop
        edge_color=edge_colors,
        edge_linewidth=2,
        bgcolor='#0a0a0a',     # Dark grey/black background
        show=False
    )
    
    # 5. Add a legend (Colorbar) so we can read the gradient
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Snow Depth (cm)', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    plt.title("Le Plateau-Mont-Royal: Simulated Snow Front", color='white', fontsize=14, pad=10)
    plt.show()

    return G

if __name__ == "__main__":
    # Ensure this matches the file you saved in the previous step
    graph_file = "data/plateau_mont_royal_drive.graphml"
    G_with_snow = plot_snow_gradient(graph_file)
    
    # Verify the feature is securely in the graph
    sample_edge = list(G_with_snow.edges(data=True))[0]
    print(f"\nVerification - Sample Edge Snow Depth: {sample_edge[2]['snow_depth']:.2f} cm")
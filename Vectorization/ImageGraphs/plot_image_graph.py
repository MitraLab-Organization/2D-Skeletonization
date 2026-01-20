import networkx as nx
import matplotlib.pyplot as plt

def plot_image_graph(g):
    """
    Plots a 2-D image graph.
    """
    if not isinstance(g, nx.Graph):
        raise TypeError("Input `g` must be a networkx Graph object.")

    # Check if the graph has 3D data
    if any('z' in g.nodes[node] for node in g.nodes()):
        raise ValueError("Plotting 3-D image graphs not supported.")

    pos = {node: (g.nodes[node]['x'], g.nodes[node]['y']) for node in g.nodes()}
    
    fig, ax = plt.subplots()
    nx.draw(g, pos, ax=ax, with_labels=False, node_size=20, node_shape='s')
    
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    return fig, ax

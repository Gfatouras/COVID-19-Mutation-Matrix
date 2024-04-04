import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix, save_npz
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg") 

print("Loading data")
mut = np.loadtxt("matrix.txt",dtype=str) # Whole mutation list
mutations = mut[:,2].astype("int") # Isolate mutation numbers
mutMtx = np.reshape(mutations, (1000,1000)) # Make list into 2D matrix
mst_matrix = minimum_spanning_tree(mutMtx) # Calculate MST
save_npz("MST.mtx",mst_matrix)

G = nx.from_scipy_sparse_matrix(mst_matrix)
np.random.seed(42)
print("Generating kamda layout (this takes a while)")
pos = nx.kamada_kawai_layout(G)

subsampled_nodes = list(G.nodes()) # set to [::x] for testing layout with less nodes

offset = (-1.979, -1.929)  # Adjust offset, edges won't line up with nodes without this
pos_shifted = {node: (pos[node][0] + offset[0], pos[node][1] + offset[1]) for node in G.nodes()}
node_sizes = {node: 50 if node == 1 else 3 for node in G.nodes}
plt.figure(figsize=(10,10)) 
print("Drawing nodes")
node_colors = {node: 'red' if node == 1 else 'deepskyblue' for node in subsampled_nodes}

nx.draw_networkx_nodes(G, pos, nodelist=subsampled_nodes,node_color=list(node_colors.values()),
node_size=list(node_sizes.values()))

print("Drawing edges")
nx.draw_networkx_edges(G, pos_shifted, width=0.5, alpha=1)
labels = {1: "NC_045512"}  # Label Root
nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='black')
plt.xlim(-1.5,1.5) # pos is usually +-2 so limit the display
plt.ylim(-1.5,1.5)

plt.title('MST ')
plt.savefig("images/MST.png")
print("Finished plot")
plt.show()
plt.close()
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#pip install _

def visualizeGraph(graphHere): #saves picture of graph to the directory the code is in (does not show edge weights atm)
    pos = nx.spring_layout(graphHere) #positions nodes for aesthetic
    nx.draw(graphHere,pos, with_labels=True, font_weight='bold',edge_cmap=plt.cm.Blues)  # ,edge_color = weight   https://matplotlib.org/tutorials/colors/colormaps.html
    plt.savefig("test.png")
    #cant quite figure out how to get edge weights to display rn
    #lead: https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_weighted_graph.html
    return

def findShortestPaths(arrayGraph,edgeWeights): #given matrix where 1s represent edges and matrix of distances between node row and node col
    g = nx.Graph()
    g.add_nodes_from(range(0,len(arrayGraph))) #adds number of nodes
    #adds edges to graph
    for x in range(0, len(arrayGraph)):
        for y in range(0, len(arrayGraph)):
            if not arrayGraph[x][y] == 0:
                g.add_edge(x,y, weight = edgeWeights[x][y])
    visualizeGraph(g)
    return (nx.all_pairs_dijkstra(g))

#main


#initiating graph rep (a) and distance for arcs (b)
a = np.zeros((10,10)) #10x10 matrix of zeroes
ad = np.zeros((10,10))
for x in range(0,10):
    for y in range(0,10):
        if not x == y:
            a[x][y] = 1
            ad[x][y] = (x+1)*(y+1)

print(ad)

allShortestPaths = findShortestPaths(a,ad)

source = 0 #doesnt matter which is which
to = 5
#second index 0 if want path length, 1 if want nodes traversed on path
print(dict(allShortestPaths)[source][0][to])
#how to use what is returned
#ref: https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra.html#networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra

#alternate way of using output
#for n, (cost, path) in allShortestPaths:
#    print("path from", n, "to", path)
#    print("cost of going from", n, "to", cost)





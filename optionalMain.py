import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#pip install _

def visualizeGraph(graphHere): #saves picture of graph to the directory the code is in (does not show edge weights atm)
    pos = nx.spring_layout(graphHere) #positions nodes for aesthetic
    #one of these two draw commands will look intuitive
    nx.draw(graphHere,pos, with_labels=True, font_weight='bold',edge_cmap=plt.cm.Blues)  # ,edge_color = weight   https://matplotlib.org/tutorials/colors/colormaps.html
    # nx.draw_circular(graphHere, with_labels=True, font_weight='bold',edge_cmap=plt.cm.Blues)

    # plt.savefig("test.png")
    plt.show()
    #cant quite figure out how to get edge weights to display rn
    #lead: https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_weighted_graph.html
    return

#How to use allpairsdjik dict

# source = 0 #doesnt matter which is which
# to = 5
# #second index 0 if want path length, 1 if want nodes traversed on path
# print(dict(allShortestPaths)[source][0][to])

def checkDistances(allpairsdjik): #check every pair of nodes for violation, return list of violations of nodes
    return

def getCost(g, allpairsdjik): #calculate the total cost using the demands and fixed costs, returns a value
    return

def displayResults(cost,distanceViolations,g):
    print("cost of graph in test.png:",cost)
    if distanceViolations is None:
        print("no distance violations")
    else:
        # for loop going through violations and printing one by one
        print("these are the violations")

    visualizeGraph(g)
    return

#goal: given array representation of a graph, display total cost and distance constraints violated
def checkGraph(arrayGraph,edgeWeights): #edgeWeights is the distance array, shouldnt be a parameter cause its the same for every run
    g = nx.Graph()
    g.add_nodes_from(range(0, len(arrayGraph)))  # adds number of nodes
    # adds edges to graph
    for x in range(0, len(arrayGraph)):
        for y in range(0, len(arrayGraph)):
            if not arrayGraph[x][y] == 0:
                g.add_edge(x, y, weight=edgeWeights[x][y])
    shortestPaths = nx.all_pairs_dijkstra(g)
    cost = getCost(g,shortestPaths)
    distanceViolations = checkDistances(shortestPaths)
    displayResults(cost,distanceViolations,g)

#main



#initiating graph rep (a) and distance for arcs (b)
a = np.zeros((14,14)) #14x14 matrix of zeroes
ad = np.zeros((14,14))
for x in range(0,14):
    for y in range(0,14):
        if not x == y:
            a[x][y] = 1
            ad[x][y] = (x+1)*(y+1)
print("distance array used:")
print(ad)
checkGraph(a,ad)

#how to use what is returned
#ref: https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra.html#networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra

#alternate way of using output
#for n, (cost, path) in allShortestPaths:
#    print("path from", n, "to", path)
#    print("cost of going from", n, "to", cost)





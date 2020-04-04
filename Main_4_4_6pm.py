import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#pip install _

def initDistances():
    df = pd.read_csv(r'Distances.csv')
    dists = np.zeros((14, 14))
    for i in range(14):
        for j in range(14):
            if j < i:
                dists[i][j] = df.iloc[i, j]
                dists[j][i] = df.iloc[i, j]
    return(dists)

def initEdges():
    counter = 0
    df = pd.read_csv(r'Edges.csv')
    dists = np.zeros((14, 14))
    for i in range(14):
        for j in range(14):
            if j < i:
                dists[i][j] = df.iloc[i, j]
                dists[j][i] = df.iloc[i, j]
                counter = counter + 1
    return (dists, counter)

def initDemand():
    df = pd.read_csv(r'Demand.csv')
    dists = np.zeros((14, 14))
    for i in range(14):
        for j in range(14):
            if j < i:
                dists[i][j] = df.iloc[i, j]
                dists[j][i] = df.iloc[i, j]
    return(dists)

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

def checkDistances(allpairsdjik, dists): #check every pair of nodes for violation, return list of violations of nodes
    x = allpairsdjik
    for i in range(14):
        for j in range(14):
            if(j != i):
                distTravelled = x[i][0][j]
                if distTravelled > (dists[i][j] * 4/3):
                    print("delay constraint violated at:",i,"to",j)
    return

def getCost(g, allpairsdjik,fixedCost,demand): #calculate the total cost using the demands and fixed costs, returns a value
    print(allpairsdjik[1][1][3])
    #also print the three highest costing arcs

    return

def displayResults(cost,g):
    print("cost of graph in test.png:",cost)

    visualizeGraph(g)
    return

#goal: given array representation of a graph, display total cost and distance constraints violated
def checkGraph(arrayGraph,edgeWeights,demand,fixedCosts): #edgeWeights is the distance array, shouldnt be a parameter cause its the same for every run
    g = nx.Graph()
    g.add_nodes_from(range(0, len(arrayGraph)))  # adds number of nodes
    # adds edges to graph
    for x in range(0, len(arrayGraph)):
        for y in range(0, len(arrayGraph)):
            if not arrayGraph[x][y] == 0:
                g.add_edge(x, y, weight=edgeWeights[x][y])
    shortestPaths = nx.all_pairs_dijkstra(g)
    shortestPaths = dict(shortestPaths)
    cost = getCost(g,shortestPaths,fixedCosts,demand)
    checkDistances(shortestPaths,edgeWeights) #will print out errors
    displayResults(cost,g)


#main
k = 0

distances = initDistances()
edges, numOfEdges = initEdges()
demand = initDemand()
#note: may have to add row of 13 , before the first row to make it accept
print(edges)


fixedCosts = numOfEdges * k

checkGraph(edges,distances,demand,fixedCosts)

#how to use what is returned
#ref: https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra.html#networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra

#alternate way of using output
#for n, (cost, path) in allShortestPaths:
#    print("path from", n, "to", path)
#    print("cost of going from", n, "to", cost)





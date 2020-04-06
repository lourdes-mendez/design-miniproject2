import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#pip install _

def putToCsv(arr,filename):
    # a = numpy.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    fn = filename + '.csv'
    np.savetxt(fn, arr, delimiter=",",fmt='%f')

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
    distMatrix = np.zeros((14,14))
    for i in range(14):
        for j in range(14):
            if(j != i and j<i):
                distTravelled = x[i][0][j]
                distMatrix[i][j] = distTravelled
                if distTravelled > (dists[i][j] * 4/3):
                    print("delay constraint violated at:",i,"to",j)
    print("demand constraints met")
    print(distMatrix)
    putToCsv(distMatrix,'lengthOfPathsTaken')
    return

def getCost(activeArr, allpairsdjik,k,demand,dist): #calculate the total cost using the demands and fixed costs, returns a value
    cost = 0
    costArr = np.zeros((14,14))
    #add ks
    for i in range(14):
        for j in range(14):
            if not i == j and j < i and activeArr[i][j] == 1:
                costArr[i][j] = k

    for i in range(14):
        for j in range(14):
            if not i == j and j < i:
                pathStuff= allpairsdjik[i][1][j]
                for x, y in zip(pathStuff[::], pathStuff[1::]):
                    if x > y:
                        costArr[x,y] += dist[x][y]*demand[i][j]
                    else:
                        costArr[y,x] += dist[x][y] * demand[i][j]

    for a in range(14):
        for b in range(14):
            cost += costArr[a][b]
    print("cost array: ")
    print(costArr)
    putToCsv(costArr,'costOfArcs')
    print("total cost for k =",k)
    print(cost)

    #also print the three highest costing arcs

    return

def displayResults(cost,g):
    print("cost of graph in test.png:",cost)
    visualizeGraph(g)
    return

#goal: given array representation of a graph, display total cost and distance constraints violated
def checkGraph(arrayGraph,edgeWeights,demand,k): #edgeWeights is the distance array, shouldnt be a parameter cause its the same for every run
    g = nx.Graph()
    g.add_nodes_from(range(0, len(arrayGraph)))  # adds number of nodes
    # adds edges to graph
    for x in range(0, len(arrayGraph)):
        for y in range(0, len(arrayGraph)):
            if not arrayGraph[x][y] == 0 and x <= y:
                if(edgeWeights[x][y] == 0 or edgeWeights[x][y] == None):
                    print("at source",x,"dest",y)
                g.add_edge(x, y, weight=edgeWeights[x][y])
    shortestPaths = nx.all_pairs_dijkstra(g)
    shortestPaths = dict(shortestPaths)

    # print(shortestPaths[1][1][0])
    # print("1 to 0 edgeweight:",edgeWeights[1][0])
    # print("1 to 0 edgeweight:", edgeWeights[0][1])
    # print(shortestPaths[1][0][4])
    # print(shortestPaths[4][0][0])

    # sp =  nx.all_pairs_dijkstra(g)
    # for n, (cost, path) in sp:
    #    print("path from", path)

    cost = getCost(arrayGraph,shortestPaths,k,demand,edgeWeights)
    checkDistances(shortestPaths,edgeWeights) #will print out errors
    # displayResults(cost,g)


#main
k = 0

distances = initDistances()
edges, numOfEdges = initEdges()
demand = initDemand()
#note: may have to add row of 13 , before the first row to make it accept
# print(edges)
# print(distances)

checkGraph(edges,distances,demand,k)

#how to use what is returned
#ref: https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra.html#networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra

#alternate way of using output
#for n, (cost, path) in allShortestPaths:
#    print("path from", n, "to", path)
#    print("cost of going from", n, "to", cost)





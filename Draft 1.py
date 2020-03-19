import numpy as np
population = np.array([8601186, 4057841, 2679044, 2359480, 1711356, 1576596, 1565929, 1453775, 1379735, 1033519, 1001104, 920984, 913939, 897536])
demand = np.zeros((15,15))
for i in range(14):
    for j in range(14):
        if population[i] < population[j]:
            demand[i,j] = demand[i,j] + [.1 * [population(i) * population(j)]**.25]
            j += 1
        else:
            j += 1
    i+=1
print(demand)

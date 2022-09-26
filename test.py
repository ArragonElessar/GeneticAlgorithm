import numpy as np

from Graph_Creator import Graph_Creator
from ROLLXYZ_FIRSTNAME import Coloring

E = 50
V = 20

gc = Graph_Creator(V)
edges = gc.CreateGraphWithRandomEdges(E)

g = np.zeros((V, V))

for edge in edges:
    s = edge[0]
    e = edge[1]
    g[s][e] = 1
    g[e][s] = 1

s1 = Coloring.generate_state()
c = Coloring(s1, g)

print(c)

"""
Program for finding minimum spanning forest using Kruskal algorithm.
Made for laboratory work 6 for discrete mathematics by a student of group 6KN-22b Ishchenko Gleb, VNTU, Ukraine.
"""


# Graph class for creating graph and finding minimum spanning forest
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []
        self.parent = [-1] * self.V

    def add_edge(self, u, v, w):
        self.graph.append([(u - 1), (v - 1), w])

    def find(self, i):
        if self.parent[i] == -1:
            return i
        return self.find(self.parent[i])

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        self.parent[x_root] = y_root

    # I use Kruskal algorithm for finding minimum spanning forest
    def kruskal_mst(self):
        result = []
        i = 0
        e = 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        while e < self.V - 1 and i < len(self.graph):
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(u)
            y = self.find(v)
            if x != y:
                e = e + 1
                result.append([(u + 1), (v + 1), w])
                self.union(x, y)
        return result

g = Graph(13) # Create graph with 13 vertices (you can change that value)

# Add graph edges below as in this example (first vertice, second vertice, edge value)
g.add_edge(1, 2, 3)
g.add_edge(1, 3, 3)
g.add_edge(1, 4, 4)
g.add_edge(2, 5, 3)
g.add_edge(2, 6, 2)
g.add_edge(3, 5, 2)
g.add_edge(3, 6, 1)
g.add_edge(3, 7, 2)
g.add_edge(4, 6, 2)
g.add_edge(4, 8, 2)
g.add_edge(5, 9, 3)
g.add_edge(5, 10, 2)
g.add_edge(6, 9, 1)
g.add_edge(6, 11, 2)
g.add_edge(7, 10, 2)
g.add_edge(7, 12, 2)
g.add_edge(8, 10, 2)
g.add_edge(8, 11, 2)
g.add_edge(8, 12, 2)
g.add_edge(9, 13, 4)
g.add_edge(10, 13, 2)
g.add_edge(11, 13, 2)
g.add_edge(12, 13, 2)

# Printing the result
result = g.kruskal_mst()
print("Minimum spanning forest consists of: ")
for edge in result:
    print(f"Edge ({edge[0]}, {edge[1]})")

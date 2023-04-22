def main():
    graph = Graph()

    while True:
        nodes = input("Enter all your nodes (all in one message, each separated by a space). Type 'exit' to exit. ")
        if nodes == 'exit':
            break
        nodes = nodes.split(' ')
        for node in nodes:
            graph.add_node(str(int(node) - 1))

    while True:
        edge = input("Enter graph edge (The input must be strictly like this: 'from_node to_node distance'). Type 'exit' to exit. ")
        if edge == 'exit':
            break
        edge = edge.split(' ')
        graph.add_edge(str(int(edge[0]) - 1), str(int(edge[1]) - 1), int(edge[2]))

    visited, paths = dijkstra(graph, '0')

    for node, distance in visited.items():
        if node != '0':
            path = get_path(paths, '0', node)
            print(f"Shortest path from 1 to {str(int(node) + 1)}: {' -> '.join([str(int(x) + 1) for x in path])} with cost {distance}")

def get_path(paths, start, end):
    path = [end]
    while path[-1] != start:
        path.append(paths[path[-1]])
    path.reverse()
    return path

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}
        self.distances = {}

    def add_node(self, value):
        self.nodes.add(value)
        self.edges[value] = []

    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance
        self.distances[(to_node, from_node)] = distance


def dijkstra(graph, start):
    visited = {start: 0}
    paths = {}

    nodes = set(graph.nodes)

    while nodes:
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node

        if min_node is None:
            break

        nodes.remove(min_node)
        current_weight = visited[min_node]

        for edge in graph.edges[min_node]:
            weight = current_weight + graph.distances[(min_node, edge)]
            if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                paths[edge] = min_node

    return visited, paths

if __name__ == "__main__":
    main()

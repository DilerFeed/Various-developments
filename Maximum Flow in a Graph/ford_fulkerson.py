from collections import defaultdict


class Graph:
    def __init__(self, graph):
        self.graph = graph
        self.rows = len(graph)

    # Метод знаходження максимального потоку у графі з допомогою алгоритму Форда-Фалкерсона
    def ford_fulkerson(self, source, sink):
        parent = [-1] * self.rows
        max_flow = 0

        # Виконуємо BFS для пошуку збільшуючих шляхів у залишковому графі
        while self.bfs(source, sink, parent):
            path_flow = float("Inf")
            s = sink

            # Знаходимо мінімальну пропускну спроможність ребер на дорозі
            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Оновлюємо пропускні здібності ребер та зворотних ребер на шляху
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

            # Збільшуємо загальний потік
            max_flow += path_flow

        return max_flow

    # BFS для пошуку збільшуючого шляху у залишковому графі
    def bfs(self, source, sink, parent):
        visited = [False] * self.rows
        queue = []

        queue.append(source)
        visited[source] = True

        while queue:
            u = queue.pop(0)

            for ind, val in enumerate(self.graph[u]):
                if visited[ind] is False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        return True if visited[sink] else False


# Ініціалізація графа
graph = [
#    1  2  3  4  5  6  7  8  9 10 11 12 13
    [0, 3, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

g = Graph(graph)

source = 0  # Вихідна вершина (на 1 менше ніж в умові!)
sink = 12    # Вхідна вершина (на 1 менше ніж в умові!)

max_flow = g.ford_fulkerson(source, sink)
print("Максимальний потік:", max_flow)

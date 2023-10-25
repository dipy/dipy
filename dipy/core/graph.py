""" A simple graph class """

class Graph:
    """ A simple graph class

    """

    def __init__(self):
        """ A graph class with nodes and edges :-)

        This class allows us to:

        1. find the shortest path
        2. find all paths
        3. add/delete nodes and edges
        4. get parent & children nodes

        Examples
        --------
        >>> from dipy.core.graph import Graph
        >>> g=Graph()
        >>> g.add_node('a',5)
        >>> g.add_node('b',6)
        >>> g.add_node('c',10)
        >>> g.add_node('d',11)
        >>> g.add_edge('a','b')
        >>> g.add_edge('b','c')
        >>> g.add_edge('c','d')
        >>> g.add_edge('b','d')
        >>> g.up_short('d')
        ['d', 'b', 'a']

        """

        self.node = {}
        self.pred = {}
        self.succ = {}

    def add_node(self, n, attr=None):
        self.succ[n] = {}
        self.pred[n] = {}
        self.node[n] = attr

    def add_edge(self, n, m, ws=True, wp=True):
        self.succ[n][m] = ws
        self.pred[m][n] = wp

    def parents(self, n):
        return self.pred[n].keys()

    def children(self, n):
        return self.succ[n].keys()

    def up(self, n):
        return self.all_paths(self.pred, n)

    def down(self, n):
        return self.all_paths(self.succ, n)

    def up_short(self, n):
        return self.shortest_path(self.pred, n)

    def down_short(self, n):
        return self.shortest_path(self.succ, n)

    def all_paths(self, graph, start, end=None, path=None):
        path = path or []
        path = path + [start]
        if start == end or graph[start] == {}:
            return [path]
        if start not in graph:
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = self.all_paths(graph, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    def shortest_path(self, graph, start, end=None, path=None):
        path = path or []
        path = path + [start]
        if graph[start] == {} or start == end:
            return path
        if start not in graph:
            return []
        shortest = None
        for node in graph[start]:
            if node not in path:
                newpath = self.shortest_path(graph, node, end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest

    def del_node_and_edges(self, n):
        try:
            del self.node[n]
        except KeyError:
            raise KeyError('node not in the graph')

        for s in self.succ[n]:
            del self.pred[s][n]
        del self.succ[n]

        for p in self.pred[n]:
            del self.succ[p][n]
        del self.pred[n]

    def del_node(self, n):
        try:
            del self.node[n]
        except KeyError:
            raise KeyError('node not in the graph')

        for s in self.succ[n]:
            for p in self.pred[n]:
                self.succ[p][s] = self.succ[n][s]
                self.pred[s][p] = self.pred[s][n]
        for s in self.succ.keys():
            try:
                del self.succ[s][n]
            except KeyError:
                pass
        for p in self.pred.keys():
            try:
                del self.pred[p][n]
            except KeyError:
                pass
        del self.succ[n]
        del self.pred[n]

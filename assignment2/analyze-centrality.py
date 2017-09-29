import sys
import snap
import matplotlib.pyplot as plt


class ShortestPathMetrics:
    """
    Stores the shortest paths between two nodes
    """
    def __init__(self, g):
        self.shortest_paths = {}
        self.__g = g
        self.__find_all(g)

    @staticmethod
    def get_key(n1, n2):
        return hash(n1.GetId()) + hash(n2.GetId())

    def add(self, n1, n2, dist):
        self.shortest_paths[self.get_key(n1, n2)] = dist

    def get(self, n1, n2):
        return self.shortest_paths[self.get_key(n1, n2)]

    def __find_all(self, g):
        print "Number of Nodes : {}".format(g.GetNodes())
        node_list = list(g.Nodes())
        node_list_2 = list(g.Nodes())
        print type(node_list)
        print type(node_list_2[0])
        count = 0

        for n1 in g.Nodes():
            for n2 in g.Nodes():
                if n1.GetId() != n2.GetId():
                    # print "finding all shortest paths between: {} - {}".format(n1.GetId(), n2.GetId())
                    dist = snap.GetShortPath(g, n1.GetId(), n2.GetId())
                    self.add(n1, n2, dist)
                count += 1
                print "count = {}".format(count)

    def get_harmonic_centrality(self, node):
        total_inverse_path_length = float(0)
        for n in self.__g.Nodes():
            if n.GetId() != node.GetId():
                total_inverse_path_length += 1/self.get(node, n)

        return total_inverse_path_length/(self.__g.GetNodes() - 1)

    def get_closeness_centrality(self, node):
        total_path_length = float(0)
        for n in self.__g:
            if n.GetId != node.GetId():
                total_path_length += self.get(node, n)

        return self.__g.GetNodes()/total_path_length if total_path_length != 0 else 0.0


def get_local_clustering_coef(g, n):
    """
    Calculate the local clustering coefficient as:
    C_i = (Number of pairs of neighbors of i that are connected) / (number of pairs of neighbors of i)

    :param g: input graph
    :param n: the node i
    :return: C_i, or 0 if degree of n is 0 or 1
    """

    neighbors = list(n.GetOutEdges())
    # print "length of neighbors {}".format(len(neighbors))
    all_pairs = []
    for i, n1 in enumerate(neighbors):
        for n2 in neighbors[i:]:
            all_pairs.append((g.GetNI(n1), g.GetNI(n2)))

    if len(all_pairs) == 0:
        print "Zero local clustering coefficient"
        return 0.0

    num_connected_neighbors = 0
    for n1, n2 in all_pairs:
        if n1.IsNbrNId(n2.GetId()):
            num_connected_neighbors += 1

    c_i = float(num_connected_neighbors) / len(all_pairs)
    # print "c_i = {}".format(c_i)
    return c_i


def write_centrality(name, centrality_list):
    print "writing file {}".format(name)
    with open(name, 'w') as f:
        for n, d in sorted(centrality_list, key=lambda t: t[1], reverse=True):
            f.write("{}, {}\n".format(n, d))
    f.close()


def main(argv):
    if len(argv) != 1:
        print "usage: python analyze-centrality.py <path/to/edgelist>"
        sys.exit(0)

    graph_file_path = argv[0]
    graph_name = graph_file_path.split('/')[-1].split('.')[0]

    print "Current file: {}".format(graph_name)

    g_snap = snap.LoadEdgeList(snap.PUNGraph, graph_file_path)

    degree = []
    closeness = []
    harmonic = []
    betweenness = []
    clustering = []

    shortest_path_metrics = ShortestPathMetrics(g_snap)

    print "Finished Setting Shortest Paths."

    for n in g_snap.Nodes():
        degree.append((n.GetId(), snap.GetDegreeCentr(g_snap, n.GetId())))
        closeness.append((n.GetId(), snap.GetClosenessCentr(g_snap, n.GetId())))
        clustering.append((n.GetId(), get_local_clustering_coef(g_snap, n)))
        harmonic.append((n.GetId(), shortest_path_metrics.get_harmonic_centrality(n)))

    print "Calculating Betweeness centrality"

    nodes_betweenness = snap.TIntFltH()
    edges_betweenness = snap.TIntPrFltH()

    snap.GetBetweennessCentr(g_snap, nodes_betweenness, edges_betweenness, 1.0)

    print "Finished calculating betweeness"

    for n in nodes_betweenness:
        betweenness.append((n, nodes_betweenness[n]))

    # Write Degree Centrality
    write_centrality(graph_name + ".degree.txt", degree)
    write_centrality(graph_name + ".closeness.txt", closeness)
    write_centrality(graph_name + "harmonic.txt", harmonic)
    write_centrality(graph_name + ".betweenness.txt", betweenness)
    write_centrality(graph_name + ".clustering.txt", clustering)


if __name__ == "__main__":
    main(sys.argv[1:])

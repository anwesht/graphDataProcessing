import sys
import snap
import networkx as nx
import matplotlib.pyplot as plt

# class ShortestPathMetrics:
#     """
#     Stores the shortest paths between two nodes
#     """
#     def __init__(self, g):
#         self.shortest_paths = {}
#         self.__g = g
#         self.__find_all(g)
#
#     @staticmethod
#     def get_key(n1, n2):
#         return hash(n1.GetId()) + hash(n2.GetId())
#
#     def add(self, n1, n2, dist):
#         self.shortest_paths[self.get_key(n1, n2)] = dist
#
#     def get(self, n1, n2):
#         return self.shortest_paths[self.get_key(n1, n2)]
#
#     def __find_all(self, g):
#         print "Number of Nodes : {}".format(g.GetNodes())
#         node_list = list(g.Nodes())
#         node_list_2 = list(g.Nodes())
#         print type(node_list)
#         print type(node_list_2[0])
#         count = 0
#
#         for n1 in g.Nodes():
#             for n2 in g.Nodes():
#                 if n1.GetId() != n2.GetId():
#                     # print "finding all shortest paths between: {} - {}".format(n1.GetId(), n2.GetId())
#                     dist = snap.GetShortPath(g, n1.GetId(), n2.GetId())
#                     self.add(n1, n2, dist)
#                 count += 1
#                 print "count = {}".format(count)
#
#     def get_harmonic_centrality(self, node):
#         total_inverse_path_length = float(0)
#         for n in self.__g.Nodes():
#             if n.GetId() != node.GetId():
#                 total_inverse_path_length += 1/self.get(node, n)
#
#         return total_inverse_path_length/(self.__g.GetNodes() - 1)
#
#     def get_closeness_centrality(self, node):
#         total_path_length = float(0)
#         for n in self.__g:
#             if n.GetId != node.GetId():
#                 total_path_length += self.get(node, n)
#
#         return self.__g.GetNodes()/total_path_length if total_path_length != 0 else 0.0
#

class ShortestPathMetrics:
    """
    Stores the shortest paths between two nodes
    """
    def __init__(self, g):
        self.__g = g
        self.shortest_path_lengths = nx.all_pairs_shortest_path_length(g)

    @staticmethod
    def get_key(n1, n2):
        return hash(n1) + hash(n2)

    def add(self, n1, n2, dist):
        self.shortest_path_lengths[self.get_key(n1, n2)] = dist

    def get(self, n1, n2):
        return self.shortest_path_lengths[self.get_key(n1, n2)]

    # def get_harmonic_centrality(self):
    #     harmonic = []
    #     total_inverse_path_length = float(0)
    #     for node, sp in self.shortest_path_lengths.items():
    #         for n, l in sp[node]:
    #             total_inverse_path_length += 1/l
    #         harmonic.append((node, total_inverse_path_length / (self.__g.number_of_nodes() - 1)))
    #     return harmonic

    def get_harmonic_centrality(self):
        """
        Calculate the harmonic centrality as in eqn 7.30 Newman:
        C'_i = (1/n-1) * sum of (1/d_ij)
        where,
            n = number of nodes in the network,
            i = node for which the centrality is being calculated,
            j = all other nodes,
            d_ij = geodesic path from i to j (length of the path)

        :param g: input graph
        :return: list of pairs with 1st entry = node i, 2nd entry = harmonic centrality of i.
        :rtype: list[tuple[int, int]]
        """
        harmonic = []
        n = nx.number_of_nodes(self.__g)

        for i in nx.nodes(self.__g):
            shortest_paths_for_node = self.shortest_path_lengths[i]
            total_inverse_path_length = 0.0

            for j, l in shortest_paths_for_node.items():
                if i != j:
                    total_inverse_path_length += 1.0/l

            harmonic_centrality = (total_inverse_path_length / (n - 1))
            harmonic.append((i, harmonic_centrality))

        return harmonic

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
    print "nx version: {}".format(nx.__version__)
    g_snap = snap.LoadEdgeList(snap.PUNGraph, graph_file_path)
    g_nx = nx.read_weighted_edgelist(path=graph_file_path, create_using=nx.Graph(), nodetype=int)

    degree = []
    closeness = []
    harmonic = []
    betweenness = []
    clustering = []

    shortest_path_metrics = ShortestPathMetrics(g_nx)

    harmonic = shortest_path_metrics.get_harmonic_centrality()  # nx
    print "Finished Setting Shortest Paths."

    print"networkx2 harmonic centrality"
    h = nx.harmonic_centrality(g_nx)

    for n, c in h.items():
        print "{} , {}".format(n, c)

    for n in g_snap.Nodes():
        degree.append((n.GetId(), snap.GetDegreeCentr(g_snap, n.GetId())))
        closeness.append((n.GetId(), snap.GetClosenessCentr(g_snap, n.GetId())))
        clustering.append((n.GetId(), get_local_clustering_coef(g_snap, n)))
        # harmonic.append((n.GetId(), shortest_path_metrics.get_harmonic_centrality(n)))    # snap



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
    write_centrality(graph_name + ".harmonic.txt", harmonic)
    write_centrality(graph_name + ".betweenness.txt", betweenness)
    write_centrality(graph_name + ".clustering.txt", clustering)


if __name__ == "__main__":
    main(sys.argv[1:])

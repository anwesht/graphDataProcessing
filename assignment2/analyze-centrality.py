import sys
import networkx as nx
import matplotlib.pyplot as plt


class ShortestPathMetrics:
    """
    Stores the shortest paths between two nodes
    """
    def __init__(self, g):
        self.__g = g
        self.shortest_path_lengths = nx.all_pairs_shortest_path_length(g)

    def get_shortest_path(self, n1, n2):
        sp = 0
        try:
            sp = self.shortest_path_lengths[n1][n2]
        except Exception as e:
            pass

        return sp

    def get_harmonic_centrality(self):
        """
        Calculate the harmonic centrality as in eqn 7.30 Newman:
        C'_i = (1/n-1) * sum of (1/d_ij)
        where,
            n = number of nodes in the network,
            i = node for which the centrality is being calculated,
            j = all other nodes,
            d_ij = geodesic path from i to j (length of the path)

        :return: dictionary with key = node i, value = harmonic centrality of i.
        :rtype: dict[int, float]
        """
        harmonic = {}
        n = nx.number_of_nodes(self.__g)

        for i in nx.nodes(self.__g):
            shortest_paths_for_node = self.shortest_path_lengths[i]
            total_inverse_path_length = 0.0

            for j, l in shortest_paths_for_node.items():
                if i != j:
                    total_inverse_path_length += 1.0/l

            harmonic_centrality = (total_inverse_path_length / (n - 1))
            harmonic[i] = harmonic_centrality

        return harmonic

    def get_closeness_centrality(self):
        """
        Calculate the closeness centrality as in eqn 7.29 Newman:
        C_i = n/sum of d_ij
        where,
            n = number of nodes in the network,
            i = node for which the centrality is being calculated,
            j = all other nodes,
            d_ij = geodesic path from i to j (length of the path)

        :return: dictionary with key = node i, value = closeness centrality of i.
        :rtype: dict[int, float]
        """
        closeness = {}
        n = nx.number_of_nodes(self.__g)

        for i in nx.nodes(self.__g):
            shortest_paths_for_node = self.shortest_path_lengths[i]
            total_path_length = 0.0

            for j, l in shortest_paths_for_node.items():
                if i != j:
                    total_path_length += 1.0 / l

            closeness_centrality = n/total_path_length
            closeness[i] = closeness_centrality

        return closeness


def get_local_clustering_coef(g, n):
    """
    Calculate the local clustering coefficient as in eqn 7.42 Newman:
    C_i = (Number of pairs of neighbors of i that are connected) / (number of pairs of neighbors of i)

    :param g: input graph
    :type g: nx.Graph
    :param n: the node i
    :type n: int
    :return: C_i, or 0 if degree of n is 0 or 1
    :rtype: float
    """
    neighbors = g.neighbors(n)
    all_pairs = []

    for i, n1 in enumerate(neighbors):
        for n2 in neighbors[i:]:
            all_pairs.append((n1, n2))

    if len(all_pairs) == 0:
        print "Zero local clustering coefficient"
        return 0.0

    num_connected_neighbors = 0.0

    for n1, n2 in all_pairs:
        if g.has_edge(n1, n2):
            num_connected_neighbors += 1

    c_i = num_connected_neighbors / len(all_pairs)

    return c_i


def get_all_local_clustering_coef(g):
    """
    Calculate local clustering coefficients for all nodes in the graph.
    :param g: Input graph
    :type g: nx.Graph
    :return: Dictionary of local clustering coefficients keyed by the node id.
    :rtype: dict[int, float]
    """
    local_cc = {}

    for n in nx.nodes(g):
        local_cc[n] = get_local_clustering_coef(g, n)

    return local_cc


def write_centrality(name, centrality_dict):
    print "writing file {}".format(name)
    with open(name, 'w') as f:
        for n, d in sorted(centrality_dict.items(), key=lambda (k, v): (v, k), reverse=True):
            f.write("{}, {}\n".format(n, d))
    f.close()


def debug(msg="Nothing to Print"):
    if DEBUG is True:
        print msg


DEBUG = True


def main(argv):
    if len(argv) != 1:
        print "usage: python analyze-centrality.py <path/to/edgelist>"
        sys.exit(0)

    graph_file_path = argv[0]
    graph_name = graph_file_path.split('/')[-1].split('.')[0]

    print "Current file: {}".format(graph_name)
    print "nx version: {}".format(nx.__version__)

    # Read in the weighted edge lists file as an undirected graph with node type integer.
    g_nx = nx.read_weighted_edgelist(path=graph_file_path, create_using=nx.Graph(), nodetype=int)

    # 1. Degree centrality
    degree = nx.degree_centrality(g_nx)
    debug("Finished degree centrality")

    # 2. Closeness Centrality
    closeness = nx.closeness_centrality(g_nx)
    debug("Finished closeness centrality")

    # 3. Harmonic Centrality
    shortest_path_metrics = ShortestPathMetrics(g_nx)
    harmonic = shortest_path_metrics.get_harmonic_centrality()
    debug("Finished harmonic centrality")

    # Uncomment to check un-normalized harmonic centrality
    # harmonic = nx.harmonic_centrality(g_nx)

    # 4. Betweenness Centrality
    # We did not normalize in class. But since we are plotting these, normalized here.
    betweenness = nx.betweenness_centrality(g_nx, normalized=True)
    debug("Finished betweenness centrality")

    # 5. Local Clustering Coefficient
    clustering = get_all_local_clustering_coef(g_nx)
    debug("Finished clustering centrality")

    # Write Degree Centrality
    write_centrality(graph_name + ".degree.txt", degree)
    write_centrality(graph_name + ".closeness.txt", closeness)
    write_centrality(graph_name + ".harmonic.txt", harmonic)
    write_centrality(graph_name + ".betweenness.txt", betweenness)
    write_centrality(graph_name + ".clustering.txt", clustering)


if __name__ == "__main__":
    main(sys.argv[1:])

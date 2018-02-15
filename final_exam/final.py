import snap
import networkx as nx
import matplotlib.pyplot as plt
import sys
from pprint import pprint

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']  # colors to use in plots.
num_figures = 0  # the number of figures drawn.
G1 = "FDS3/G1.edgelist"
G2 = "FDS3/G2.edgelist"
G3 = "FDS3/G3.edgelist"
G4 = "FDS3/G4.edgelist"
G5 = "FDS3/dnc-corecipient-weigthed.edges"

class Metrics:
    def __init__(self, path, is_weighted=False):
        self.name = path.split('/')[-1].split('.')[0]
        self.g_nx = self.__nx_graph(path, is_weighted)
        self.g_snap = self.__snap_graph(path)
        self.num_nodes = 0
        self.num_edges = 0
        self.clustering_coeff = 0.0
        self.num_triads = 0
        self.transitivity = 0.0
        self.avg_degree = 0.0
        self.avg_spl = 0
        self.diameter = 0.0

    @staticmethod
    def __snap_graph(path):
        return snap.LoadEdgeList(snap.PUNGraph, path)

    @staticmethod
    def __nx_graph(path, is_weighted):
        if is_weighted:
            return nx.read_weighted_edgelist(path)
        else:
            return nx.read_edgelist(path)

    def __str__(self):
        return "Name of Graph: {}\n".format(self.name) \
               + "\tNumber of Nodes: {}\n".format(self.num_nodes) \
               + "\tNumber of Edges: {}\n".format(self.num_edges) \
               + "\tClustering Coefficient: {}\n".format(self.clustering_coeff) \
               + "\tNumber of Triads: {}\n".format(self.num_triads) \
               + "\tNumber of Transitivity: {}\n".format(self.transitivity) \
               + "\tAverage Degree: {}\n".format(self.avg_degree) \
               + "\tAverage Path Length: {}\n".format(self.avg_spl) \
               + "\tDiameter: {}\n".format(self.diameter)

    def calculate_spl(self):
        """
        Calculates the average shortest path length.
        If the network is disconnected, calculates for the largest connected component
        :return: avg_spl
        """
        try:
            avg_spl = nx.average_shortest_path_length(self.g_nx)
        except nx.NetworkXError as e:
            print "{}: calculating spl for largest connected component.".format(e)
            avg_spl = nx.average_shortest_path_length(max(nx.connected_component_subgraphs(self.g_nx), key=len))

        return avg_spl

    def calculate_basic(self):
        self.num_nodes = nx.number_of_nodes(self.g_nx)
        self.num_edges = nx.number_of_edges(self.g_nx)

        # Calculate the average degree
        print "calculating average degree"
        sum_degrees = 0.0
        for _, d in nx.degree(self.g_nx):
            sum_degrees += d

        self.avg_degree = sum_degrees / self.num_nodes

        return self

    def calculate(self):
        """
        Calculates the metrics on the network using both networkx and snap
        :return: self
        """
        self.calculate_basic()

        print "calculating clustering coefficient."
        self.clustering_coeff = snap.GetClustCf(self.g_snap, -1)

        print "calculating transitivity"
        self.transitivity = nx.transitivity(self.g_nx)

        print "calculating triads"
        self.num_triads = snap.GetTriads(self.g_snap, -1)

        print "calculating diameter"
        self.diameter = snap.GetBfsFullDiam(self.g_snap, 150, False)

        print "calculating spl"
        self.avg_spl = self.calculate_spl()

        return self


def get_degree_dict(g):
    """
    Get the degree distribution of the given graph
    :param g: the input graph
    :type g: Graph
    :return: a dictionary with key = degree and value = it's distribution
    :rtype: dict [int, float]
    """
    degree_dict = {}
    for n in g.nodes():
        d = g.degree(n)
        if d not in degree_dict:
            degree_dict[d] = 0
        degree_dict[d] += 1
    degree_dict = sorted(degree_dict.items())
    return degree_dict


def plot_degree_distribution(name, grouped_degree_dict, log_scale=False):
    """
    Generate a log-log plot of degree distribution(Pk) vs degree(k)
    :param name: Name of the graph
    :type name: String
    :param grouped_degree_dict: dictionary with key = label and value = dictionary of degree distribution
    :type grouped_degree_dict: dict[str, dict[int, float]]
    :param log_scale: boolean to select log scale
    :type log_scale: boolean
    """
    global num_figures

    fig = plt.figure(num_figures)
    num_figures += 1

    fig.suptitle("Plot of the degree distribution \nfor: {}".format(name))

    ax = fig.add_subplot(111)
    if log_scale is True:
        ax.set_xscale('log')
        ax.set_yscale('log')
        name = "log-log_{}".format(name)

    ax.set_xlabel("Degree")  # Degree
    ax.set_ylabel("Occurrence")  # frequency/ degree distribution

    i = 0
    for (key, degree_dict) in grouped_degree_dict.items():
        degree_dict = sorted(degree_dict.items())
        x_axis_vals = [k for (k, _) in degree_dict]
        y_axis_vals = [v for (_, v) in degree_dict]

        ax.scatter(x_axis_vals, y_axis_vals, color=COLORS[i], label=key, marker=".")

        i = (i + 1) % len(COLORS)

    ax.legend(loc='best')
    fig.savefig(name)


def draw(g, name="graph"):
    nx.draw_circular(g)
    plt.savefig("{}.png".format(name), format="PNG")


def connected_components():
    # g = nx.read_edgelist("data/as-caida20040105.txt")
    g = nx.read_edgelist("data/random5000by6.txt")
    # g = nx.complete_graph(100)
    cc = nx.connected_components(g)
    cc_list = [c for c in sorted(cc, key=len, reverse=True)]
    for c in cc_list:
        print c
        print "size: {}".format(len(c))
        break

    # draw(g, "g")

    print "G: number of nodes = {}. edges = {}".format(g.number_of_nodes(), g.number_of_edges())

    gc = nx.complement(g, "gc")
    print "GC: number of nodes = {}. edges = {}".format(gc.number_of_nodes(), gc.number_of_edges())

    cc = nx.connected_components(gc)
    cc_list = [c for c in sorted(cc, key=len, reverse=True)]
    for c in cc_list:
        print c
        print "size: {}".format(len(c))
        # break

    draw(gc, "gc")


def main():
    # Snap: read graph
    # g_snap = snap.LoadEdgeList(snap.PUNGraph, '<FILEPATH>')

    # Networkx: read graph
    g_nx_1 = nx.read_edgelist(G1)
    g_nx_2 = nx.read_edgelist(G2)
    g_nx_3 = nx.read_edgelist(G3)
    g_nx_4 = nx.read_edgelist(G4)

    # Networkx: read weighted graph
    # g_nx_w = nx.read_weighted_edgelist('<FILEPATH>')

    # print (Metrics(G1).calculate())
    # print (Metrics(G2).calculate())
    # print (Metrics(G3).calculate())
    # print (Metrics(G4).calculate())

    # Plotting degree distribution
    grouped_degree_dict = dict()
    # grouped_degree_dict['G1'] = dict(get_degree_dict(g_nx_1))
    # grouped_degree_dict['G2'] = dict(get_degree_dict(g_nx_2))
    # grouped_degree_dict['G3'] = dict(get_degree_dict(g_nx_3))
    # grouped_degree_dict['G4'] = dict(get_degree_dict(g_nx_4))

    # Separate plots for degree distribution
    # g1_degree_dict = {'G1': dict(get_degree_dict(g_nx_1))}
    # g2_degree_dict = {'G2': dict(get_degree_dict(g_nx_2))}
    # g3_degree_dict = {'G3': dict(get_degree_dict(g_nx_3))}
    # g4_degree_dict = {'G4': dict(get_degree_dict(g_nx_4))}
    #
    # plot_degree_distribution(G1.split('/')[-1].split('.')[0] + ".png", g1_degree_dict)
    # plot_degree_distribution(G1.split('/')[-1].split('.')[0] + ".png", g1_degree_dict, True)
    #
    # plot_degree_distribution(G2.split('/')[-1].split('.')[0] + ".png", g2_degree_dict)
    # plot_degree_distribution(G2.split('/')[-1].split('.')[0] + ".png", g2_degree_dict, True)
    #
    # plot_degree_distribution(G3.split('/')[-1].split('.')[0] + ".png", g3_degree_dict)
    # plot_degree_distribution(G3.split('/')[-1].split('.')[0] + ".png", g3_degree_dict, True)
    #
    # plot_degree_distribution(G4.split('/')[-1].split('.')[0] + ".png", g4_degree_dict)
    # plot_degree_distribution(G4.split('/')[-1].split('.')[0] + ".png", g4_degree_dict, True)

    print dict(get_degree_dict(g_nx_3))
    nx.write_gexf(g_nx_1, "g1.gexf")
    nx.write_gexf(g_nx_2, "g2.gexf")
    # nx.write_gexf(g_nx_3, "g3.gexf")
    nx.write_gexf(g_nx_4, "g4.gexf")


def main3():
    g = nx.read_weighted_edgelist(G5)
    # print (Metrics(G5, is_weighted=True).calculate())

    # g_degree_dict = {'G4': dict(get_degree_dict(g))}
    #
    # plot_degree_distribution(G4.split('/')[-1].split('.')[0] + ".png", g_degree_dict)
    # plot_degree_distribution(G4.split('/')[-1].split('.')[0] + ".png", g_degree_dict, True)

    # cc = nx.connected_components(g)
    # cc_list = [c for c in sorted(cc, key=len, reverse=True)]
    # for c in cc_list:
    #     print c
    #     print "size: {}".format(len(c))

    # Centralities:
    eigen_vector = nx.eigenvector_centrality(g)

    # print eigen_vector
    eigen = sorted(eigen_vector.iteritems(), key=lambda (k,v): (v,k), reverse=True)
    pprint("eigenvector:")
    pprint(eigen)

    eigen_vector_numpy = nx.eigenvector_centrality_numpy(g)
    print("\n\neigen_vector_numpy:")
    pprint(sorted(eigen_vector_numpy.iteritems(), key=lambda (k,v): (v,k), reverse=True))

    closeness = nx.closeness_centrality(g)
    print("\n\ncloseness:")
    pprint(sorted(closeness.iteritems(), key=lambda (k,v): (v,k), reverse=True))

    betweenness = nx.betweenness_centrality(g)
    print("\n\nbetweenness")
    pprint(sorted(betweenness.iteritems(), key=lambda (k,v): (v,k), reverse=True))


def centrality():
    g = nx.read_weighted_edgelist(G5, create_using=nx.DiGraph())

    nx.write_gexf(g, "g5.gexf")

    cent = dict()

    for node in g.nodes():
        sum = 0.0
        num = 0
        total = 0
        for n in g.neighbors(node):
            total += 1
            if g.has_edge(node, n):
                weight = g[node][n]['weight']
                sum += weight
                num += 1

        if num > 0:
            cent[node] = (sum/num, num, total)
        else:
            cent[node] = (0.0, 0, 0)

    pprint(sorted(cent.iteritems(), key=lambda (k,v): (v,k), reverse=True))


if __name__ == "__main__":
    # main()
    # main3()
    # connected_components()
    centrality()

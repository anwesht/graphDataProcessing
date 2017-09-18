import sys
import networkx as nx
from random import choice
import matplotlib.pyplot as plt
import numpy
import snap

RANDOM_SIZE_LIST = [10, 100, 1000]


# def plot_deg_distribution(g):
#


def generate_graph_nx(num_nodes):
    """
    Generates a complete graph and removes nodes that are not divisible by 2 and 3.
    Writes the corresponding edgelist to a file: random5000by6.txt
    :return: graph
    """
    gen_graph = nx.complete_graph(num_nodes)

    for n in range(gen_graph.number_of_nodes()):
        if not (n % 2 == 0 and n % 3 == 0):
            gen_graph.remove_node(n)

    nx.write_edgelist(gen_graph, "random5000by6.txt", data=False)


def main(argv):
    if len(argv) != 1:
        print "usage: python gen-structure.py <path/to/edgelist>"
        sys.exit(0)

    # Q0. Uncomment to generate random5000by6.txt edge list.
    # generate_graph_nx(5000)

    graph_file_path = argv[0]
    graph_file_name = graph_file_path.split('/')[-1]

    print "Current file: {}".format(graph_file_name)

    g_nx = nx.read_edgelist(graph_file_path)

    # Q1.a. print the number of nodes in the graph
    print "Number of nodes in {}: {}".format(graph_file_name, g_nx.number_of_nodes())
    # Q1.b print the number of edges in the graph
    print "Number of edges in {}: {}".format(graph_file_name, g_nx.number_of_edges())

    # Q2.a. nx.degree returns a number or a dictionary with nodes as keys and degree as value.
    degree_dict = nx.degree(g_nx)
    nodes_with_degree_1 = filter(lambda k: degree_dict[k] == 1, degree_dict.keys())
    print "Number of nodes with degree = 1 in {}: {}".format(graph_file_name, len(nodes_with_degree_1))

    # Q2.b. find max degree.
    max_degree = 0
    nodes_with_max_degree = []

    for k, v in degree_dict.items():
        if v > max_degree:
            max_degree = v
            nodes_with_max_degree = [k]
        elif v == max_degree:
            nodes_with_max_degree.append(k)

    print "Max Degree is {}".format(max_degree)
    # print "Max Degree is {}".format(sorted(degree_dict.values())[-1])  # sanity check
    print "Node id(s) with highest degree in {}: {}".format(graph_file_name,
                                                            ", ".join(str(i) for i in nodes_with_max_degree))

    # Q2.c.
    for node in nodes_with_degree_1:
        neighbors = g_nx.neighbors(node)
        avg_degree_n_2 = float(0)
        num_of_n_2 = 0

        for n_1 in neighbors:
            n_2 = g_nx.neighbors(n_1)
            avg_degree_n_2 = reduce(lambda acc, d: acc + g_nx.degree(d), n_2, 0)
            num_of_n_2 += len(n_2)

        print "The average degree of {}'s 2-hop neighborhood is: {}".format(node, avg_degree_n_2/num_of_n_2)

    g_snap = snap.LoadEdgeList(snap.PUNGraph, graph_file_path)

    # Q2.d Plot the degree distribution
    snap.PlotOutDegDistr(g_snap, graph_file_name+"-degree_distribution", "Plot of the degree distribution")
    print "Degree distribution of {} is in: {}".format(graph_file_name,
                                                       "outDeg."+graph_file_name+"-degree_distribution.png")

    # Q3.a. Approximate full diameter (maximum shortest path length)
    full_diameters = []
    for max_size in RANDOM_SIZE_LIST:
        full_diam = snap.GetBfsFullDiam(g_snap, max_size, False)
        full_diameters.append(full_diam)
        print "Approx. diameter in {} with sampling {} nodes: {}".format(graph_file_name,
                                                                         max_size, full_diam)

    print "Approx. diameter in {} (mean and variance): {}, {}.".format(graph_file_name,
                                                                       numpy.mean(full_diameters),
                                                                       numpy.var(full_diameters))

    # Q3.b. Effective Diameter
    effective_diameters = []
    for max_size in RANDOM_SIZE_LIST:
        effective_diam = snap.GetBfsEffDiam(g_snap, max_size, False)
        effective_diameters.append(effective_diam)

        print "Approx. effective diameter in {} with sampling {} nodes: {}".format(graph_file_name,
                                                                                   max_size, effective_diam)

    print "Approx. effective diameter in {} (mean and variance): {}, {}.".format(graph_file_name,
                                                                                 numpy.mean(effective_diameters),
                                                                                 numpy.var(effective_diameters))

    # Q3.c. Plot distribution of shortest path lengths
    snap.PlotShortPathDistr(g_snap, graph_file_name+"-shortest_path_distribution",
                            "Plot of the distribution of shortest path lengths")
    print "Shortest path distribution of {} is in: {}".format(graph_file_name,
                                                              "diam."+graph_file_name+"-shortest_path_distribution.png")

    # Q4.a. Fraction of nodes in the largest connected component.
    num_nodes_largest_comp_gnx = 0
    connected_components_gnx = sorted(nx.connected_components(g_nx), key=len, reverse=True)

    if len(connected_components_gnx) > 0:
        num_nodes_largest_comp_gnx = len(connected_components_gnx[0])

    frac_largest_comp_gnx = num_nodes_largest_comp_gnx/g_nx.number_of_nodes()

    print "Fraction of nodes in largest connected component in {}: {}".format(graph_file_name, frac_largest_comp_gnx)

    # Q4.b. Fraction of nodes in the largest connected component of the complement of the real graph
    num_nodes_largest_comp_gnxc = 0
    g_nx_c = nx.complement(g_nx, "g_nx_c")
    connected_components_gnxc = sorted(nx.connected_components(g_nx_c), key=len, reverse=True)

    if len(connected_components_gnxc) > 0:
        num_nodes_largest_comp_gnxc = len(connected_components_gnxc[0])

    frac_largest_comp_gnxc = num_nodes_largest_comp_gnxc / g_nx_c.number_of_nodes()

    print "Fraction of nodes in largest connected component in {}'s complement: {}".format(graph_file_name,
                                                                                           frac_largest_comp_gnxc)

    # todo Q4.c. Plot of the distribution of sizes of connected components.
    comp_size_freq = {}
    for conn in connected_components_gnx:
        comp_size = len(conn)
        # print comp_size
        if comp_size not in comp_size_freq:
            comp_size_freq[comp_size] = 0

        comp_size_freq[comp_size] += 1

    comp_size_freq = sorted(comp_size_freq.items())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_axis_vals = [k for (k, _) in comp_size_freq]
    y_axis_vals = [v for (_, v) in comp_size_freq]
    ax.plot(x_axis_vals, y_axis_vals, 'bo')
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.title("Plot of the distribution of sizes of connected components \n for {}".format(graph_file_name))
    plt.xlabel("Component Size")
    plt.ylabel("Frequency")

    plt.draw()
    plt.show()


    # g_snap_c = snap.TUNGraph.New()
    #
    # for n in g_snap.Nodes():
    #     g_snap_c.AddNode(int(n.GetId()))
    #
    # for s, d in g_nx_c.edges():
    #     # print s + ", " + d
    #     g_snap_c.AddEdge(int(s), int(d))
    #
    # snap.PlotSccDistr(g_snap, graph_file_name+"-scc_distribution",
    #                   "Plot of the distribution of sizes of connected components")
    # print "Component size distribution of {} is in: ".format(graph_file_name,
    #                                                          graph_file_name + "-scc_distribution")


    # snap.PlotSccDistr(g_snap_c, graph_file_name + "_complement-size_of_connected_comps_distribution",
    #                   "Plot of the distribution of sizes of connected components")
    # print "Component size distribution of the complement of {} is in: {}".format(
    #     graph_file_name,
    #     graph_file_name + "_complement-size_of_connected_comps_distribution")




    # R.add_edges_from(((n, n2)
    #                   for n, nbrs in G.adjacency_iter()
    #                   for n2 in G if n2 not in nbrs
    #                   if n != n2))

    # snap.GenRndGnm(snap.PNGraph, 100, 1000)

    # nx.draw(G)
    # plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])

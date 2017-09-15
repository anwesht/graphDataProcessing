import sys
import networkx as nx
from random import choice
import matplotlib.pyplot as plt
import numpy
import snap

RANDOM_SIZE_LIST = [10, 100, 1000]


def main(argv):
    if len(argv) != 1:
        print "usage: python gen-structure.py <path/to/edgelist>"
        sys.exit(0)

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
    max_degree, nodes_with_max_degree = reduce(
        lambda (md, max_ids), key:
            (md, max_ids) if degree_dict[key] < md
            else (md, max_ids.append(key)) if degree_dict[key] == md
            else (degree_dict[key], [key]),
        degree_dict,
        (0, list())
    )

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

    # todo Q2.d. plot the degree distribution

    # Q3.a. Approximate full diameter (maximum shortest path length)
    # full_diameters = []
    # for max_size in RANDOM_SIZE_LIST:
    #     full_diam = 0
    #     for _ in range(0, max_size, 2):
    #         n1 = choice(g_nx.nodes())
    #         n2 = choice(g_nx.nodes())
    #         shortest_path_length = nx.shortest_path_length(g_nx, n1, n2)
    #         if full_diam < shortest_path_length:
    #             full_diam = shortest_path_length
    #
    #     full_diameters.append(full_diam)
    #     print "Approx. diameter in {} with sampling {} nodes: {}".format(graph_file_name,
    #                                                                      max_size, full_diam)
    #
    # print "Approx. diameter in {} (mean and variance): {}, {}.".format(graph_file_name,
    #                                                                    numpy.mean(full_diameters),
    #                                                                    numpy.var(full_diameters))

    g_snap = snap.LoadEdgeList(snap.PUNGraph, graph_file_path)

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

    # todo Q3.c. Plot distribution of shortest path lengths

    # Q4.a. Fraction of nodes in the largest connected component.
    num_nodes_largest_comp_gnx = 0
    connected_components_gnx = sorted(nx.connected_components(g_nx), key=len, reverse=True)

    if len(connected_components_gnx) > 0:
        num_nodes_largest_comp = len(connected_components_gnx[0])

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


    # snap.GenRndGnm(snap.PNGraph, 100, 1000)

    # nx.draw(G)
    # plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])

import sys
import networkx as nx
from random import choice
import matplotlib.pyplot as plt
import numpy
import snap

RANDOM_SIZE_LIST = [10, 100, 1000]


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


def setup_log_log_plot(ax, x_axis_vals, y_axis_vals):
    """
    Sets up the axis values for a log-log plot.
    :param ax:
    :param x_axis_vals: list of values for x-axis
    :param y_axis_vals: list of values for y-axis
    :return: void
    """
    ax.set_xscale('log')
    ax.set_yscale('log')
    if len(y_axis_vals) == 1:
        lo = float(y_axis_vals[0]) / 10
        hi = float(y_axis_vals[0]) * 10
        ax.set_ylim(ymin=lo, ymax=hi)
    if len(x_axis_vals) == 1:
        lo = float(x_axis_vals[0]) / 10
        hi = float(x_axis_vals[0]) * 10
        ax.set_xlim(xmin=lo, xmax=hi)


def plot_degree_distribution(g, graph_file_name):
    """
    Counts the frequency of each degree and plots it in a log-log plot.
    :param g: graph for which the degree distribution is to be plotted.
    :param graph_file_name: current edgelist name
    :return: void
    """
    degree_dict = {}

    for n in g.nodes():
        d = g.degree(n)
        if d not in degree_dict:
            degree_dict[d] = 0
        degree_dict[d] += 1

    degree_dict = sorted(degree_dict.items())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_axis_vals = [k for (k, _) in degree_dict]
    y_axis_vals = [v for (_, v) in degree_dict]

    setup_log_log_plot(ax, x_axis_vals, y_axis_vals)

    plt.title("Plot of the degree distribution for \n for {}".format(graph_file_name))
    plt.xlabel("Degree")
    plt.ylabel("Frequency")

    ax.plot(x_axis_vals, y_axis_vals, '-rx')

    # Uncomment to view plt
    # plt.show()
    fig.savefig(graph_file_name + "-degree_distribution.png")


def plot_distribution_of_connected_components(connected_components_gnx, graph_file_name):
    """
    Counts the frequency of the size of connected components and plots it in a log-log plot using matplotlib
    :param connected_components_gnx: list of connected components of the graph
    :param graph_file_name: name of the original graph
    :return: void
    """
    comp_size_freq = {}
    for conn in connected_components_gnx:
        comp_size = len(conn)
        if comp_size not in comp_size_freq:
            comp_size_freq[comp_size] = 0

        comp_size_freq[comp_size] += 1
    comp_size_freq = sorted(comp_size_freq.items())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_axis_vals = [k for (k, _) in comp_size_freq]
    y_axis_vals = [v for (_, v) in comp_size_freq]

    setup_log_log_plot(ax, x_axis_vals, y_axis_vals)

    plt.title("Plot of the distribution of sizes of connected components \n for {}".format(graph_file_name))
    plt.xlabel("Component Size")
    plt.ylabel("Frequency")

    ax.plot(x_axis_vals, y_axis_vals, '-bx')

    # Uncomment to view plt
    # plt.show()
    fig.savefig(graph_file_name + "-scc_distribution.png")


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
    # print "Check: Max Degree is {}".format(sorted(degree_dict.values())[-1])  # sanity check
    print "Node id(s) with highest degree in {}: {}".format(graph_file_name,
                                                            ", ".join(str(i) for i in nodes_with_max_degree))

    # Q2.c. 2-hop Neighbors
    for node in nodes_with_degree_1:
        neighbors = g_nx.neighbors(node)  # nodes in 1 hop. Should just be 1.
        if len(neighbors) > 1:  # Sanity check
            print "Not a node with degree 1!!!"
            continue

        n1 = neighbors[0]

        n2s = g_nx.neighbors(n1)
        sum_degrees_n2s = reduce(lambda acc, d: acc + g_nx.degree(d), n2s, 0)
        avg_degree_n2 = float(sum_degrees_n2s)/len(n2s)

        print "The average degree of {}'s 2-hop neighborhood is: {}".format(node, avg_degree_n2)

    # Using snap for plots.
    g_snap = snap.LoadEdgeList(snap.PUNGraph, graph_file_path)

    # Q2.d Plot the degree distribution
    # snap.PlotOutDegDistr(g_snap, graph_file_name+"-degree_distribution", "Plot of the degree distribution")
    plot_degree_distribution(g_nx, graph_file_name)
    print "Degree distribution of {} is in: {}".format(graph_file_name,
                                                       graph_file_name+"-degree_distribution.png")

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

    frac_largest_comp_gnx = float(num_nodes_largest_comp_gnx)/g_nx.number_of_nodes()

    print "Fraction of nodes in largest connected component in {}: {}".format(graph_file_name, frac_largest_comp_gnx)

    # Q4.b. Fraction of nodes in the largest connected component of the complement of the real graph
    num_nodes_largest_comp_gnxc = 0
    g_nx_c = nx.complement(g_nx, "g_nx_c")
    # Sort the connected components based on size
    connected_components_gnxc = sorted(nx.connected_components(g_nx_c), key=len, reverse=True)

    if len(connected_components_gnxc) > 0:
        num_nodes_largest_comp_gnxc = len(connected_components_gnxc[0])

    frac_largest_comp_gnxc = float(num_nodes_largest_comp_gnxc) / g_nx_c.number_of_nodes()

    print "Fraction of nodes in largest connected component in {}'s complement: {}".format(graph_file_name,
                                                                                           frac_largest_comp_gnxc)

    # Q4.c. Plot of the distribution of sizes of connected components.
    plot_distribution_of_connected_components(connected_components_gnx, graph_file_name)

    print "Component size distribution of {} is in: {}".format(graph_file_name,
                                                               graph_file_name + "-scc_distribution.png")

    plot_distribution_of_connected_components(connected_components_gnxc,
                                              graph_file_name + "_complement")

    print "Component size distribution of the complement of {} is in: {}".format(
        graph_file_name,
        graph_file_name + "_complement-scc_distribution.png")


if __name__ == "__main__":
    main(sys.argv[1:])

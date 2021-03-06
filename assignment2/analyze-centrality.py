import sys
import networkx as nx
import matplotlib.pyplot as plt
from random import random
import re


DEBUG = False  # Toggles debug messages


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
        Calculate the closeness centrality as in:
        https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html
        C_i = ((n - 1)/(N-1)) * (n-1) / sum of d_ij
        where,
            n = number of nodes in the network,
            i = node for which the centrality is being calculated,
            j = all other nodes,
            d_ij = geodesic path from i to j (length of the path)

        Using eqn 7.29 Newman:[ C_i = n/sum of d_ij ] results in 0.0 centrality due to disconnected components.

        :return: dictionary with key = node i, value = closeness centrality of i.
        :rtype: dict[int, float]
        """
        closeness = {}
        N = nx.number_of_nodes(self.__g)

        for i in nx.nodes(self.__g):
            shortest_paths_for_node = self.shortest_path_lengths[i]
            n = len(shortest_paths_for_node)
            # Newman's formula yields in 0 centrality.
            # if len(shortest_paths_for_node) != n:
            #     closeness[i] = 0.0
            #     continue

            total_path_length = 0.0

            for j, l in shortest_paths_for_node.items():
                if i != j:
                    total_path_length += l

            # closeness_centrality = n/total_path_length  # Newman's formula
            closeness_centrality = 0.0

            if total_path_length > 0.0:
                closeness_centrality = ((n - 1.0)/(N - 1.0)) * ((n - 1.0) / total_path_length)

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
        if i < len(neighbors):
            for n2 in neighbors[i:]:
                all_pairs.append((n1, n2))

    if len(all_pairs) == 0:
        print "Zero local clustering coefficient"
        return 0.0

    num_connected_neighbors = 0.0

    for n1, n2 in all_pairs:
        if g.has_edge(n1, n2):
            num_connected_neighbors += 1.0

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


def is_equal(f1, f2):
    precision = 0.000001  # considering upto 6 decimal places.
    # precision = 0.00000000000001  # considering upto 6 decimal places.
    # precision = 0.0001
    return abs(f1 - f2) < precision


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


def plot_centrality_from_file(file_path=None, is_log_plot=False):
    if file_path is None:
        print "No input file provided."
        return

    graph_name = file_path.split('/')[-1].split('.')[0]
    centrality_name = file_path.split('/')[-1].split('.')[1]

    debug("Plotting centrality %s for %s" % (graph_name, centrality_name))

    ranked_centralities = {}
    prev_centrality = random()
    rank = 0

    with open(file_path, 'r') as f:
        for line in f:
            l = line.strip().split(",")
            if len(l) == 0:
                break

            centrality = float(l[1].strip())
            if not is_equal(centrality, prev_centrality):
                rank += 1
                ranked_centralities[rank] = centrality

    ranked_centralities = sorted(ranked_centralities.items())
    x_axis_vals = [k for (k, _) in ranked_centralities]
    y_axis_vals = [v for (_, v) in ranked_centralities]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    xlabel = "Rank"
    ylabel = "%s centrality" % centrality_name

    if is_log_plot is True:
        setup_log_log_plot(ax, x_axis_vals, y_axis_vals)
        xlabel = "log( " + xlabel + " )"
        ylabel = "log( " + ylabel + " )"

    plt.title("Plot of the {} for \n for {}".format(centrality_name, graph_name))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.plot(x_axis_vals, y_axis_vals, '-r')

    file_name = graph_name + "." + centrality_name + (".log-log.png" if is_log_plot else ".png")
    print "filename: {}".format(file_name)
    fig.savefig(file_name)


def debug(msg="Nothing to Print"):
    if DEBUG is True:
        print msg


def plot_centralities(argv, is_log_plot=False):
    file_name = argv[0].split('/')[-1].split('.')[0]

    plot_centrality_from_file(file_name + ".degree.txt", is_log_plot=is_log_plot)
    plot_centrality_from_file(file_name + ".closeness.txt", is_log_plot=is_log_plot)
    plot_centrality_from_file(file_name + ".harmonic.txt", is_log_plot=is_log_plot)
    plot_centrality_from_file(file_name + ".betweenness.txt", is_log_plot=is_log_plot)
    plot_centrality_from_file(file_name + ".clustering.txt", is_log_plot=is_log_plot)


def read_centrality_values_from_file(fp):
    vals = []
    with open(fp, 'r') as f:
        for line in f:
            l = line.strip().split(",")
            if len(l) == 0:
                break

            val = float(l[1].strip())
            vals.append(val)

    return vals


def scatter_plot(fp1, fp2):
    if fp1 is None or fp2 is None:
        print "No files provided for scatter plot."
        return

    gn1 = fp1.split('/')[-1].split('.')[0]
    cn1 = fp1.split('/')[-1].split('.')[1]

    gn2 = fp2.split('/')[-1].split('.')[0]
    cn2 = fp2.split('/')[-1].split('.')[1]

    debug("Scatter Plot %s-%s VS %s-%s" % (gn2, cn2, gn1, cn1))

    c1_vals = read_centrality_values_from_file(fp1)
    c2_vals = read_centrality_values_from_file(fp2)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    xlabel = gn1 + " - " + cn1
    ylabel = gn2 + " - " + cn2

    plt.title("Scatter Plot {}-{} VS {}-{}".format(gn2, cn2, gn1, cn1))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.scatter(c1_vals, c2_vals, marker=".", color="r")

    file_name = ylabel + "_vs_" + xlabel + ".png"
    print "filename: {}".format(file_name)
    fig.savefig(file_name)


def plot_scatter_matrix(argv):
    file_name = argv[0].split('/')[-1].split('.')[0]
    centralities = [
        file_name + ".degree.txt",
        file_name + ".closeness.txt",
        file_name + ".harmonic.txt",
        file_name + ".betweenness.txt",
        file_name + ".clustering.txt"
    ]

    for i, c1 in enumerate(centralities):
        if i < len(centralities):
            for j, c2 in enumerate(centralities[i+1:]):
                print c1 + "   VS   " + c2
                scatter_plot(c2, c1)


def get_top_entities_from_file(file_path, name_file_path, top=10):
    if file_path is None:
        print "No input file provided."
        return

    print file_path

    graph_name = file_path.split('/')[-1].split('.')[0]
    centrality_name = file_path.split('/')[-1].split('.')[1]

    print "---------------------------------------------------------------------"
    print "Top {} entities by {} centrality for {}".format(top, centrality_name, graph_name)
    print "---------------------------------------------------------------------"

    name_dict = {}
    header = ""
    with open(name_file_path, 'r') as n:
        header = n.readline()
        for line in n:
            node_id = int(line.split()[0])
            # node_name = line.split()[1]  # Use if the name list file is separated by space without ""
            node_name = re.search('".*"', line).group(0)  # Use if the name list has the node names enclosed in ""
            name_dict[node_id] = node_name

    i = 1
    output_file = "{}.{}.top-{}.txt".format(graph_name, centrality_name, top)

    with open(file_path, 'r') as f, open(output_file, 'w') as o:
        o.write(header+"\n")
        for line in f:
            node_id = int(line.strip().split(",")[0])
            o.write("{}. {} => {}\n".format(i, node_id, name_dict[node_id]))
            print "{}. {} => {}".format(i, node_id, name_dict[node_id])
            if i >= top:
                break
            i += 1


def get_top_entities_by_centrality(argv, top=10):
    graph_file_path = argv[0]
    graph_name = graph_file_path.split('/')[-1].split('.')[0]
    name_file_path = argv[1]

    data_file_path = graph_name

    get_top_entities_from_file(data_file_path + ".degree.txt", name_file_path, top)
    get_top_entities_from_file(data_file_path + ".closeness.txt", name_file_path, top)
    get_top_entities_from_file(data_file_path + ".harmonic.txt", name_file_path, top)
    get_top_entities_from_file(data_file_path + ".betweenness.txt", name_file_path, top)


def main(argv):
    graph_file_path = argv[0]
    graph_name = graph_file_path.split('/')[-1].split('.')[0]

    print "Current file: {}".format(graph_name)
    print "nx version: {}".format(nx.__version__)

    # Read in the weighted edge lists file as an undirected graph with node type integer.
    # g_nx = nx.read_weighted_edgelist(path=graph_file_path, create_using=nx.Graph(), nodetype=int)
    g_nx = nx.read_weighted_edgelist(path=graph_file_path, create_using=nx.Graph())

    # 1. Degree centrality
    degree = nx.degree_centrality(g_nx)
    debug("Finished degree centrality")

    # 2. Closeness Centrality
    shortest_path_metrics = ShortestPathMetrics(g_nx)
    closeness = shortest_path_metrics.get_closeness_centrality()
    debug("Finished closeness centrality")

    # 3. Harmonic Centrality
    harmonic = shortest_path_metrics.get_harmonic_centrality()
    debug("Finished harmonic centrality")

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
    if len(sys.argv) == 2:
        main(sys.argv[1:])
    elif len(sys.argv) == 3 and sys.argv[2] == "-plt":
        plot_centralities(sys.argv[1:], is_log_plot=False)
    elif len(sys.argv) == 3 and sys.argv[2] == "-scatter":
        plot_scatter_matrix(sys.argv[1:])
    elif len(sys.argv) == 3:
        get_top_entities_by_centrality(sys.argv[1:])
    elif len(sys.argv) == 4 and sys.argv[2] == "-plt" and sys.argv[3] == "-log":
        plot_centralities(sys.argv[1:], is_log_plot=True)
    else:
        print "usage: python analyze-centrality.py <path/to/edgelist> [-plt [-log]| path/to/names/list]"
        sys.exit(0)

import sys
import networkx as nx
import matplotlib.pyplot as plt
from random import choice


DEBUG = True


def get_local_clustering_coef(g, n):
    """
    This code was duplicated here from analyze-centrality.py because of the naming conventions used in the files.
    file named with a hyphen "-" cannot be imported here.
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


def debug(msg="Nothing to Print"):
    if DEBUG is True:
        print msg


def plot_distribution(distribution_dict, xlabel, ylabel, graph_name):
    distribution_dict = sorted(distribution_dict.items())

    x_axis_vals = [k for (k, _) in distribution_dict]
    y_axis_vals = [v for (_, v) in distribution_dict]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.title("Plot of the {} vs {} for \n for {}".format(xlabel, ylabel, graph_name))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.plot(x_axis_vals, y_axis_vals, '-r')

    file_name = graph_name + "." + xlabel + "_vs_" + ylabel + ".png"
    fig.savefig(file_name)

    return file_name


def main(argv):
    if len(argv) != 1:
        print "usage: python analyze-centrality.py <path/to/edgelist>"
        sys.exit(0)

    graph_file_path = argv[0]
    graph_name = graph_file_path.split('/')[-1].split('.')[0]

    print "Current file: {}".format(graph_name)
    print "nx version: {}".format(nx.__version__)

    # 1. Read in the weighted edge lists file as an undirected graph with node type integer.
    # g_nx = nx.read_weighted_edgelist(path=graph_file_path, create_using=nx.Graph(), nodetype=int)
    g_nx = nx.read_weighted_edgelist(path=graph_file_path, create_using=nx.Graph())

    # Number of triads.
    num_triads = len(nx.triangles(g_nx))
    print "Number of triads: {}".format(num_triads)

    # 2. Local Clustering coefficient of a randomly selected node
    random_node = choice(nx.nodes(g_nx))
    lcc = get_local_clustering_coef(g_nx, random_node)
    print "Clustering coefficient of random node {} in {}: {}".format(random_node, graph_name, lcc)

    # 3. Number of triads a random node participates in
    random_node = choice(nx.nodes(g_nx))
    num_triads_for_rand_node = nx.triangles(g_nx, random_node)
    print "Number of triads node {} participates in {} triads".format(random_node, num_triads_for_rand_node)

    # 4. Watts-Strogratz (average over local) and global clustering coefficients.
    avg_clustering = nx.average_clustering(g_nx)
    global_clustering_dict = nx.clustering(g_nx)
    avg_global_clustering = sum(v for k, v in global_clustering_dict.items()) / len(global_clustering_dict)
    print "Clustering coefficient of the network: {} (Watts-Strogatz); {} (global)".format(avg_clustering,
                                                                                           avg_global_clustering)

    # 5. Plot of the k-core edge-size distribution
    k_core_edge_dict = {}
    k_core_node_dict = {}
    max_num_cores = max(nx.core_number(g_nx).values())
    for i in range (0, max_num_cores):
        k_core_edge_dict[i] = nx.k_core(g_nx, k=i).number_of_edges()
        k_core_node_dict[i] = nx.k_core(g_nx, k=i).number_of_nodes()

    file_name = plot_distribution(k_core_edge_dict, "core k", "number of edges in k-core", graph_name)

    print "k-core edge-size distribution is in: {}".format(file_name)

    # 6. Plot of the k-core node-size distribution
    file_name = plot_distribution(k_core_edge_dict, "core k", "number of nodes in k-core", graph_name)

    print "k-core node-size distribution is in: {}".format(file_name)


if __name__ == "__main__":
    main(sys.argv[1:])
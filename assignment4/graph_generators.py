import snap
import networkx as nx
import matplotlib.pyplot as plt
import sys


GRAPHS = [
    "data/USairport_2010.elist.txt",
    # "data/imdb_actor.elist.txt"
]

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']  # colors to use in plots.
num_figures = 0  # the number of figures drawn.


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


def generate_graphs():
    for path in GRAPHS:
        name = path.split('/')[-1].split('.')[0]

        metrics = Metrics(path, True).calculate_basic()

        print metrics

        # Generate Erdos-Renyi (Random) Graph
        # args: type, num_nodes, num_edges
        er = snap.GenRndGnm(snap.PNGraph, metrics.num_nodes, metrics.num_edges)
        snap.SaveEdgeList(er, "{}_er.elist".format(name))

        # Generate Watts-Strogatz (Small World) Graph
        # args: num_nodes, node_out_degree (average out degree will be twice this value, rewire_prob)
        ws = snap.GenSmallWorld(metrics.num_nodes, int(metrics.avg_degree)/2, 0.2)
        snap.SaveEdgeList(ws, "{}_ws.elist".format(name))

        # Generate Barabasi-Albert model (scale-free with preferential attachment) Graph
        # args: (num_nodes, degree of each node desired)
        ba = snap.GenPrefAttach(metrics.num_nodes, int(metrics.avg_degree)/2)
        snap.SaveEdgeList(ba, "{}_ba.elist".format(name))

        # Generate Forest Fire model Graph
        # args: (num_nodes, forward_prob, backward_prob)
        if name == "USairport_2010":
            ff = snap.GenForestFire(metrics.num_nodes, 0.3599, 0.3599)  # Selected value for US Airports data-set
            snap.SaveEdgeList(ff, "{}_ff.elist".format(name))

            ff = snap.GenForestFire(int(metrics.num_nodes / 10), 0.3599, 0.3599)
            snap.SaveEdgeList(ff, "{}_ffdiv10.elist".format(name))

            ff = snap.GenForestFire(metrics.num_nodes * 10, 0.3599, 0.3599)
            snap.SaveEdgeList(ff, "{}_ffx10.elist".format(name))
        else:
            ff = snap.GenForestFire(metrics.num_nodes, 0.3467, 0.3467)   # selected
            snap.SaveEdgeList(ff, "{}_ff.elist".format(name))

            ff = snap.GenForestFire(int(metrics.num_nodes/10), 0.3467, 0.3467)
            snap.SaveEdgeList(ff, "{}_ffdiv10.elist".format(name))

            ff = snap.GenForestFire(metrics.num_nodes*10, 0.3467, 0.3467)
            snap.SaveEdgeList(ff, "{}_ffx10.elist".format(name))


def calculate_metrics():
    """
    Calculate the metrics on the generated graphs.
    Note: Requires the edge lists of all the graphs.
    :return: void
    """
    for path in GRAPHS:
        name = path.split('/')[-1].split('.')[0]

        print "***** {}: Generated Graphs *****".format(name)

        print (Metrics(path, is_weighted=True).calculate())

        er_name = "{}_er.elist".format(name)
        print (Metrics(er_name).calculate())

        ws_name = "{}_ws.elist".format(name)
        print (Metrics(ws_name).calculate())

        ba_name = "{}_ba.elist".format(name)
        print (Metrics(ba_name).calculate())

        ff_name = "{}_ff.elist".format(name)
        print (Metrics(ff_name).calculate())

        ff_name = "{}_ffdiv10.elist".format(name)
        print (Metrics(ff_name).calculate())

        # Uncomment to calculate metrics for the X10 graph.
        # ff_name = "{}_ffx10.elist".format(name)
        # print (Metrics(ff_name).calculate())

        print "***********************************"


def plot():
    """
    PLot the degree distribution on the generated graphs.
    Note: Requires the edge lists of all the graphs.
    :return: void
    """
    for path in GRAPHS:
        all_degree_dict = dict()
        name = path.split('/')[-1].split('.')[0]

        print "***** Potting Degree Distribution for: {} *****".format(name)

        original = nx.read_weighted_edgelist(path)
        all_degree_dict[name] = dict(get_degree_dict(original))

        er_name = "output/{}_er.elist".format(name)
        er = nx.read_edgelist(er_name)
        all_degree_dict[er_name.split('/')[-1]] = dict(get_degree_dict(er))

        ws_name = "output/{}_ws.elist".format(name)
        ws = nx.read_edgelist(ws_name)
        all_degree_dict[ws_name.split('/')[-1]] = dict(get_degree_dict(ws))

        ba_name = "output/{}_ba.elist".format(name)
        ba = nx.read_edgelist(ba_name)
        all_degree_dict[ba_name.split('/')[-1]] = dict(get_degree_dict(ba))

        ff_name = "output/{}_ff.elist".format(name)
        ff = nx.read_edgelist(ff_name)
        all_degree_dict[ff_name.split('/')[-1]] = dict(get_degree_dict(ff))

        # Uncomment the following to include in the plot.
        # ff_name = "output/{}_ffx10.elist".format(name)
        # ffx = nx.read_edgelist(ff_name)
        # all_degree_dict[ff_name.split('/')[-1]] = dict(get_degree_dict(ffx))
        #
        # ff_name = "output/{}_ffdiv10.elist".format(name)
        # ffdiv = nx.read_edgelist(ff_name)
        # all_degree_dict[ff_name.split('/')[-1]] = dict(get_degree_dict(ffdiv))

        plot_degree_distribution(name+".png", all_degree_dict)
        plot_degree_distribution(name+".png", all_degree_dict, True)


def main(argv):
    if len(argv) == 1 and argv[0] == "plt":
        plot()
    elif len(argv) == 1 and argv[0] == "metrics":
        calculate_metrics()
    else:
        generate_graphs()


if __name__ == "__main__":
    main(sys.argv[1:])

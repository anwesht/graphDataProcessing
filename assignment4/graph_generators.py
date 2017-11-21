import snap
import networkx as nx


GRAPHS = [
    "data/imdb_actor.elist.txt",
    "data/USairport_2010.elist.txt"
]


class Metrics:
    def __init__(self, g, name):
        self.name = name
        self.g = g
        self.num_nodes = 0
        self.num_edges = 0
        self.clustering_coeff = 0.0
        self.num_triads = 0
        self.transitivity = 0.0
        self.avg_degree = 0.0
        self.avg_spl = 0
        self.diameter = 0.0

    def __str__(self):
        return "Name of Graph: {}\n".format(self.name) \
               + "\tNumber of Nodes: {}\n".format(self.num_nodes) \
               + "\tNumber of Edges: {}\n".format(self.num_edges) \
               + "\tClustering Coefficient: {}\n".format(self.clustering_coeff) \
               + "\tNumber of Triads: {}\n".format(self.num_triads) \
               + "\tNumber of Transitivity: {}\n".format(self.transitivity) \
               + "\tAverage Degree: {}\n".format(self.avg_degree) \
               + "\tAverage Path Length: {}\n".format(self.avg_spl)

    def calculate_spl(self):
        try:
            avg_spl = nx.average_shortest_path_length(self.g)
        except nx.NetworkXError as e:
            print "{}: calculating spl for largest connected component.".format(e)
            avg_spl = nx.average_shortest_path_length(max(nx.connected_component_subgraphs(self.g), key=len))

        return avg_spl

    def calculate(self):
        self.num_nodes = nx.number_of_nodes(self.g)
        self.num_edges = nx.number_of_edges(self.g)

        # Calculate the average degree
        sum_degrees = 0.0
        for _, d in nx.degree(self.g):
            sum_degrees += d

        self.avg_degree = sum_degrees / self.num_nodes

        self.clustering_coeff = nx.clustering(self.g)
        self.transitivity = nx.transitivity(self.g)
        self.diameter = nx.diameter(self.g)

        self.avg_spl = self.calculate_spl()

        return self


def main():
    # Check parameters for graphs
    for path in GRAPHS:
        name = path.split('/')[-1].split('.')[0]
        # g = snap.LoadEdgeList(snap.PUNGraph, path)
        # metrics = Metrics(g, name).calculate()

        g = nx.read_weighted_edgelist(path)
        metrics = Metrics(g, name).calculate()

        print metrics

        # Generate Erdos-Renyi (Random) Graph
        # args: type, num_nodes, num_edges
        er = snap.GenRndGnm(snap.PNGraph, metrics.num_nodes, metrics.num_edges)
        snap.SaveEdgeList(er, "erdos-renyi-{}".format(name))

        # Generate Watts-Strogatz (Small World) Graph
        # args: num_nodes, node_out_degree (average out degree will be twice this value, rewire_prob)
        ws = snap.GenSmallWorld(metrics.num_nodes, int(metrics.avg_degree), 0.5)
        snap.SaveEdgeList(ws, "watts-strogatz-{}".format(name))

        # Generate Barabasi-Albert model (scale-free with preferential attachment) Graph
        # args: (num_nodes, degree of each node desired)
        ba = snap.GenPrefAttach(metrics.num_nodes, int(metrics.avg_degree))
        snap.SaveEdgeList(ba, "barabasi-albert-{}".format(name))

        # Generate Forest Fire model Graph
        # args: (num_nodes, forward_prob, backward_prob)
        ff = snap.GenForestFire(metrics.num_nodes, 0.2, 0.2)
        snap.SaveEdgeList(ff, "forest-fire-{}".format(name))

        print "----------"

    for path in GRAPHS:
        name = path.split('/')[-1].split('.')[0]

        er_name = "erdos-renyi-{}".format(name)
        er = nx.read_edgelist(er_name)
        print (Metrics(er, er_name).calculate())

        ws_name = "watts-strogatz-{}".format(name)
        ws = nx.read_edgelist(ws_name)
        print (Metrics(ws, ws_name).calculate())

        ba_name = "barabasi-albert-{}".format(name)
        ba = nx.read_edgelist(ba_name)
        print (Metrics(ba, ba_name).calculate())

        ff_name = "forest-fire-{}".format(name)
        ff = nx.read_edgelist(ff_name)
        print (Metrics(ff, ff_name).calculate())


if __name__ == "__main__":
    main()

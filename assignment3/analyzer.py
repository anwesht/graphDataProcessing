import networkx as nx


class Analyzer:
    def __init__(self, graph, graph_name="No Name"):
        self.graph = graph
        self.graph_name = graph_name
        self.assortativity = float('inf')
        self.num_nodes = float('inf')
        self.num_edges = float('inf')
        self.avg_degree = float('inf')

    def __str__(self):
        return self.graph_name + ": " \
               + "Number of Nodes: {} ".format(self.num_nodes) \
               + "Number of Edges: {} ".format(self.num_edges) \
               + " Assortativity: {}".format(self.assortativity) \
               + " Average Degree: {}".format(self.avg_degree)

    def get_avg_degree(self):
        sum_degrees = 0.0
        for n in self.graph.nodes():
            sum_degrees += self.graph.degree(n)

        return sum_degrees/self.graph.number_of_nodes()

    def analyze(self):
        self.assortativity = nx.degree_assortativity_coefficient(self.graph)
        self.num_nodes = self.graph.number_of_nodes()
        self.num_edges = self.graph.number_of_edges()
        self.avg_degree = self.get_avg_degree()
        print (self)

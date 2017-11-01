import networkx as nx
import matplotlib.pyplot as plt


class Analyzer:
    colors = ['r', 'b', 'g']

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

    def get_assortativity(self):
        return nx.degree_assortativity_coefficient(self.graph)

    def plot_degree_distribution(self, *graphs):
        """
        Counts the frequency of each degree and plots it in a log-log plot.
        :param g: graph for which the degree distribution is to be plotted.
        :param graph_file_name: current edgelist name
        :return: void
        """
        degree_dict_list = [self.get_degree_dict(self.graph)]

        for g in graphs:
            degree_dict_list.append(self.get_degree_dict(g))

        plt.title("Plot of the degree distribution for \n for {}".format(self.graph_name))
        plt.xlabel("Degree")
        plt.ylabel("Frequency")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')

        i = 0
        for degree_dict in degree_dict_list:
            x_axis_vals = [k for (k, _) in degree_dict]
            y_axis_vals = [v for (_, v) in degree_dict]

            ax.plot(x_axis_vals, y_axis_vals, color=self.colors[i])
            i += 1

        # Uncomment to view plt
        # plt.show()
        # fig.savefig(self.graph_name+ "-degree_distribution.png")

    # @staticmethod
    def get_degree_dict(self):
        degree_dict = {}
        for n in self.graph.nodes():
            d = self.graph.degree(n)
            if d not in degree_dict:
                degree_dict[d] = 0
            degree_dict[d] += 1
        degree_dict = sorted(degree_dict.items())
        sum_degree = sum(map(lambda (k, v): v, degree_dict), 0)

        self.avg_degree = float(sum_degree)/self.graph.number_of_nodes()

        degree_dict = map(lambda (k, v): (k, float(v)/self.graph.number_of_nodes()), degree_dict)
        return degree_dict

    def get_avg_degree(self):
        sum_degrees = 0.0
        for n in self.graph.nodes():
            sum_degrees += self.graph.degree(n)

        return sum_degrees/self.graph.number_of_nodes()

    def analyze(self):
        self.assortativity = self.get_assortativity()
        self.num_nodes = self.graph.number_of_nodes()
        self.num_edges = self.graph.number_of_edges()
        # self.avg_degree = nx.average_degree_connectivity(self.graph)
        self.avg_degree = self.get_avg_degree()
        print (self)

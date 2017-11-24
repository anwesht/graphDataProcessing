import networkx as nx
from numpy.random import choice as rand
from matplotlib import pyplot as plt


num_figures = 0
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']  # colors to use in plots.


class Simulator:
    MAX_STEPS = 1000

    def __init__(self, g, model):
        """
        Simulate the contagion model.
        :param model: contagion model to use
        :type model: SIRModel
        :param g: graph to simulate on
        :type g: nx.Graph
        """
        self.model = model
        self.g = model.init_graph(g)
        self.steps_taken = 0
        self.infected = model.init_infection(g)
        self.contagion_stats = {self.steps_taken: self.infected}

    def step(self):
        self.steps_taken += 1
        new_infections = set()

        for n in self.g.nodes():
            new_infections.update(self.model.apply(n, self.g))

        self.contagion_stats[self.steps_taken] = new_infections
        self.infected = new_infections

    def run(self, steps_to_take=-1):
        if steps_to_take == -1:
            steps_to_take = self.MAX_STEPS

        for _ in range(steps_to_take):
            self.step()
            if len(self.infected) == 0:
                print 'No more infections after step: {}'.format(self.steps_taken)
                break

        return self.contagion_stats


class SIRModel:
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2
    LENGTH_OF_INFECTION = 'lof'
    PROB_INFECTION = 'prob_infection'
    STATE = 'state'

    def __init__(self, prob_infection=0.5, length_of_infection=1, reinfection_factor=0.5, num_seeds=1):
        self.prob_infection = prob_infection
        self.length_of_infection = length_of_infection
        self.reinfection_factor = reinfection_factor
        self.num_seeds = num_seeds

    def init_graph(self, graph):
        nx.set_node_attributes(graph, self.SUSCEPTIBLE, self.STATE)
        nx.set_node_attributes(graph, self.prob_infection, self.PROB_INFECTION)
        nx.set_node_attributes(graph, 0, self.LENGTH_OF_INFECTION)

        return graph
        # return self.init_infection(graph)

    def init_infection(self, graph):
        infected = set(rand(graph.nodes(), size=self.num_seeds))
        for n in infected:
            graph.node[n][self.STATE] = self.INFECTED

        print "Init infection: {}".format(infected)
        # return graph
        return infected

    def apply(self, node, graph):
        def update_infected_list(n):
            attr = graph.node[n]
            if attr[self.STATE] == self.INFECTED:
                infected.add(n)

        def update_susceptible(n):
            attr = graph.node[n]
            attr[self.STATE] = rand([self.INFECTED, self.SUSCEPTIBLE],
                                    p=[attr[self.PROB_INFECTION], 1 - attr[self.PROB_INFECTION]])
            update_infected_list(n)

        def update_infected(n):
            attr = graph.node[n]
            # print "Updating infected"
            if attr[self.LENGTH_OF_INFECTION] == self.length_of_infection:
                attr[self.STATE] = self.RECOVERED
                attr[self.LENGTH_OF_INFECTION] = 0
                attr[self.PROB_INFECTION] *= self.reinfection_factor
            else:
                attr[self.LENGTH_OF_INFECTION] += 1

                # iterate over adjacent nodes of node in graph.
                for neighbor in graph.neighbors(node):
                    a = graph.node[neighbor]
                    # print "neighbor : {}, attr: {}".format(neighbor, a)
                    if a[self.STATE] == self.SUSCEPTIBLE:
                        update_susceptible(neighbor)
                    elif a[self.STATE] == self.RECOVERED:
                        update_recovered(neighbor)
            update_infected_list(n)

        def update_recovered(n):
            attr = graph.node[n]
            attr[self.STATE] = rand([self.INFECTED, self.RECOVERED],
                                    p=[attr[self.PROB_INFECTION], 1 - attr[self.PROB_INFECTION]])
            update_infected_list(n)

        infected = set()

        node_attr = graph.node[node]

        # print "Node: {}, attr: {}".format(node, node_attr)
        if node_attr[self.STATE] == self.INFECTED:
            # print "Infected Node: {}".format(node_attr)
            update_infected(node)
        elif node_attr[self.STATE] == self.RECOVERED:
            # print "Recovered Node: {}".format(node_attr)
            update_recovered(node)

        return infected


def plot(name, grouped_dict, log_scale=False):
    global num_figures

    fig = plt.figure(num_figures)
    num_figures += 1

    fig.suptitle("Plot of infections over time \nfor: {}".format(name))

    ax = fig.add_subplot(111)
    if log_scale is True:
        ax.set_xscale('log')
        ax.set_yscale('log')
        name = "log-log_{}".format(name)

    ax.set_xlabel("Time")  # Degree
    ax.set_ylabel("Number of Infections")  # frequency/ degree distribution

    i = 0
    for (key, infections_dict) in grouped_dict.items():
        infections_dict = sorted(infections_dict.items())
        x_axis_vals = [k for (k, _) in infections_dict]
        y_axis_vals = [len(v) for (_, v) in infections_dict]

        ax.plot(x_axis_vals, y_axis_vals, color=COLORS[i], label=key, marker=".")

        i = (i + 1) % len(COLORS)

    ax.legend(loc='best')
    fig.savefig(name)


def main():
    g = nx.erdos_renyi_graph(500, 0.2)

    sir = SIRModel(prob_infection=0.02, length_of_infection=2, reinfection_factor=0.5, num_seeds=2)

    simulator = Simulator(g, sir)
    simulator.run()

    contagion_stats_dict = {'erdos-renyi': simulator.contagion_stats}

    g = nx.watts_strogatz_graph(500, 50, 0.2)
    contagion_stats_dict['watts-strogatz'] = Simulator(g, sir).run()

    plot('SIRModel', contagion_stats_dict)


if __name__ == "__main__":
    main()

import networkx as nx
import sys
from numpy.random import choice as rand
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod
from pprint import pprint


class EpidemicModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def init_graph(self, graph):
        raise NotImplementedError(
            'Function init_graph() should initialize the graph with attributes required by the model.')

    @abstractmethod
    def init_infection(self, graph, selection_strategy):
        raise NotImplementedError('Function init_infection should infect some seed nodes.')

    @abstractmethod
    def apply(self, node, graph, visited, step):
        raise NotImplementedError('Function apply() should update the input node according to the model.')


num_figures = 0
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']  # colors to use in plots.


class Simulator:
    MAX_STEPS = 1000

    def __init__(self, g, model, name="graph", seed_selection='rand'):
        """
        Simulate the contagion model.
        :param model: contagion model to use
        :type model: EpidemicModel
        :param g: graph to simulate on
        :type g: nx.Graph
        :param name: the name of the graph
        :type name: str
        """
        self.name = name
        self.model = model
        self.g = model.init_graph(g)
        self.steps_taken = 0.0
        self.infected = model.init_infection(g, selection_strategy=seed_selection)
        # print "Number of initial Infection: {}".format(len(self.infected))
        self.contagion_stats = {self.steps_taken: set(self.infected)}
        self.has_infection = True
        self.metrics = dict()

    def step(self):
        """
        Execute one step of the simulation
        :return: void
        """
        self.steps_taken += 1
        new_infections = set()

        visited = set()
        self.has_infection = False

        for n in self.g.nodes():
            if n not in visited:
                visited, has_infection = self.model.apply(node=n, graph=self.g, visited=visited, step=self.steps_taken)
                self.has_infection |= has_infection
            if self.g.node[n]['state'][-1][0] == 1:
                self.has_infection = True
                self.infected.add(n)
                new_infections.add(n)

        self.contagion_stats[self.steps_taken] = new_infections

    def run(self, steps_to_take=-1):
        """
        Executes the simulation.
        :param steps_to_take: number of steps to simulate.
                              if -1, simulates till the epidemic dies off.
        :return: self
        """
        if steps_to_take == -1:
            steps_to_take = self.MAX_STEPS

        for i in range(steps_to_take):
            self.step()
            if self.has_infection is False:
                print 'No more infections after step: {}'.format(self.steps_taken)
                break

        self.write_graph(0)
        self.calculate_metrics()
        # return self.contagion_stats
        return self

    def calculate_metrics(self):
        """
        Calculate the metrics on epidemic model
        :return: void
        """
        length_of_epidemic = self.steps_taken
        num_nodes_infected = len(self.infected)
        time_of_max_infection, max_infections_at_a_time = max([(t, len(v)) for t, v in self.contagion_stats.items()],
                                                            key=lambda tup: tup[1])

        self.metrics = {
            'length_of_epidemic': length_of_epidemic,
            'num_nodes_infected': num_nodes_infected,
            'time_of_max_infection': time_of_max_infection,
            'max_infections_at_a_time': max_infections_at_a_time,
            'num_nodes': self.g.number_of_nodes(),
            'num_edges': self.g.number_of_edges()
        }

    def write_graph(self, timestep=0):
        nx.write_gexf(self.g, "{}-{}.gexf".format(self.name, timestep))


class SIRModel(EpidemicModel):
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2
    LENGTH_OF_INFECTION = 'lof'
    PROB_INFECTION = 'prob_infection'
    STATE = 'state'
    IS_INFECTED = 'is_infected'

    def __init__(self, prob_infection=0.5, length_of_infection=1, reinfection_factor=0.5, num_seeds=1):
        """
        Initialize the SIR Model.
        :param prob_infection: Probability that a susceptible node is infected by an infected neighbour.
        :param length_of_infection: The number of time steps the infection lasts
        :param reinfection_factor: The factor by which the probability of infection decreases after
                                   the previous infection.
        :param num_seeds: The number of nodes to infect at the beginning.
        """
        self.prob_infection = prob_infection
        self.length_of_infection = length_of_infection
        self.reinfection_factor = reinfection_factor
        self.num_seeds = num_seeds

    def init_graph(self, graph):
        """
        Initialize the graph with node attributes required to track the changes of
        node attributes over time.
        :param graph: The graph on which to apply the SIRModel
        :type graph: nx.Graph
        :return: a set of infected nodes.
        :rtype: set[int]
        """
        for n in graph.nodes():
            state = list()
            state.append((self.SUSCEPTIBLE, 0.0, None))

            # print "Initial state: {}".format(state)
            graph.node[n][self.STATE] = state
            graph.node[n][self.PROB_INFECTION] = self.prob_infection
            graph.node[n][self.LENGTH_OF_INFECTION] = 0

        return graph

    def init_infection(self, graph, selection_strategy):
        """
        Initialize infections
        :param graph: The graph on which to apply the SIRModel
        :type graph: nx.Graph
        :param selection_strategy: the strategy to use to select infected nodes
        :return: a set of infected nodes.
        :rtype: set[int]
        """
        if selection_strategy == 'max_deg':
            return self.init_infection_degree(graph, is_max=True)
        elif selection_strategy == 'min_deg':
            return self.init_infection_degree(graph, is_max=False)
        else:
            return self.init_infection_rand(graph)

    def init_infection_rand(self, graph):
        """
        Initialize random nodes as infected.
        :param graph: The graph on which to apply the SIRModel
        :type graph: nx.Graph
        :return: a set of infected nodes.
        :rtype: set[int]
        """
        infected = set(rand(graph.nodes(), size=self.num_seeds))
        for n in infected:
            graph.node[n][self.STATE][-1] = (self.INFECTED, 0.0, None)
            # print "Infecting: {}".format(n)
            # print graph.node[n][self.STATE]
        return infected

    def init_infection_degree(self, graph, is_max=True):
        """
        Initialize nodes as infected based on degree
        :param graph: The graph on which to apply the SIRModel
        :type graph: nx.Graph
        :param is_max: boolean to pick minimum or maximum degree
        :return: a set of infected nodes.
        :rtype: set[int]
        """
        def get_index(index):
            if is_max:
                return -1-index
            else:
                return index

        degree_dict = nx.degree(graph)
        degree_dict = sorted(dict(degree_dict).iteritems(), key=lambda (k, v): (v, k))

        # print degree_dict
        infected = set()
        for i in range(self.num_seeds):
            n, _ = degree_dict[get_index(i)]
            infected.add(n)

            graph.node[n][self.STATE][-1] = (self.INFECTED, 0.0, None)
            # print "Infecting: {} with degree: {}".format(n, graph.degree(n))
            # print graph.node[n][self.STATE]

        return infected

    def apply(self, node, graph, visited, step=0):
        def update_visited_list(n):
            """
            Update a set of visited nodes. This set is used to keep track of the nodes already
            visited in this time step, so that it is not updated multiple times.
            :param n: the node to add to visited set
            :return: True if n is infected otherwise False
            :rtype: boolean
            """
            visited.add(n)
            attr = graph.node[n]
            if attr[self.STATE][-1][0] == self.INFECTED:
                return True
            else:
                return False

        def update_susceptible(n):
            """
            Calculate the new state of a susceptible node.
            :param n: the susceptible node to update
            :return: void
            """
            attr = graph.node[n]
            new_state = int(rand([self.INFECTED, self.SUSCEPTIBLE],
                            p=[attr[self.PROB_INFECTION], 1 - attr[self.PROB_INFECTION]]))

            if new_state == self.INFECTED:
                # Update the end time for previous state
                last_state = attr[self.STATE][-1]
                attr[self.STATE][-1] = (last_state[0], last_state[1], step - 1)
                # Add a new state
                attr[self.STATE].append((self.INFECTED, step, None))

        def update_infected(n):
            """
            Calculate the new state of the infected node n.
            Also simulate the infection of neighbouring susceptible nodes.
            :param n: infected node to update
            :return: a set of newly infected nodes
            :rtype: set[int]
            """
            attr = graph.node[n]
            new_infection = False
            if attr[self.LENGTH_OF_INFECTION] == self.length_of_infection:
                # Update the end time for previous state
                last_state = attr[self.STATE][-1]
                attr[self.STATE][-1] = (last_state[0], last_state[1], step-1)
                # Add a new state
                attr[self.STATE].append((self.RECOVERED, step, None))
                # Update other attributes
                attr[self.LENGTH_OF_INFECTION] = 0
                attr[self.PROB_INFECTION] *= self.reinfection_factor
            else:
                attr[self.LENGTH_OF_INFECTION] += 1
                # iterate over adjacent nodes of node in graph.
                for neighbor in graph.neighbors(node):
                    a = graph.node[neighbor]
                    # print "current node: {}, neighbor : {}, attr: {}".format(n, neighbor, a)
                    if a[self.STATE][-1][0] == self.SUSCEPTIBLE:
                        update_susceptible(neighbor)
                    elif a[self.STATE][-1][0] == self.RECOVERED and neighbor not in visited:
                        # check to avoid recovery and reinfection in same time step.
                        update_recovered(neighbor)

                    new_infection |= update_visited_list(neighbor)

            return new_infection

        def update_recovered(n):
            """
            Calculate the new state of recovered nodes.
            :param n: recovered node to update
            :return: void
            """
            attr = graph.node[n]
            new_state = int(rand([self.INFECTED, self.RECOVERED],
                                 p=[attr[self.PROB_INFECTION], 1 - attr[self.PROB_INFECTION]]))
            if new_state == self.INFECTED:
                # Update the end time for previous state
                last_state = attr[self.STATE][-1]
                attr[self.STATE][-1] = (last_state[0], last_state[1], step-1)
                # Add a new state
                attr[self.STATE].append((self.INFECTED, step, None))

        has_infection = False
        node_attr = graph.node[node]

        if node_attr[self.STATE][-1][0] == self.INFECTED:
            has_infection |= update_infected(node)
        elif node_attr[self.STATE][-1][0] == self.RECOVERED:
            update_recovered(node)

        has_infection |= update_visited_list(node)

        return visited, has_infection


def plot(name, grouped_dict, log_scale=False):
    """
    Plots the number of infections vs time graph
    :param name: Name of the graph
    :param grouped_dict: a dictionary of dictionaroes with data to plot
    :param log_scale: boolean to use log scale.
    :return: void
    """
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
    fig.savefig("{}.png".format(name), format="PNG")


def draw(g, name="graph"):
    nx.draw(g)
    plt.savefig("{}.png".format(name), format="PNG")


def generate_graphs():
    """
    Generates the graphs to use for the experiments.
    :return: void
    """
    num_nodes = 100
    er = nx.erdos_renyi_graph(n=num_nodes, p=0.07)  # Graph generated using these parameters
    nx.write_edgelist(er, 'er-avg_deg-6.elist.txt')
    ws = nx.watts_strogatz_graph(n=num_nodes, k=6, p=0.2)  # Graph generated using these parameters
    nx.write_edgelist(ws, 'ws-avg_deg-6.elist.txt')
    pc = nx.powerlaw_cluster_graph(num_nodes, m=3, p=0.45)  # Graph generated using these parameters
    nx.write_edgelist(pc, 'pc-avg_deg-6.elist.txt')


def experiment():
    """
    Simulate the SIR epidemic model for a number of parameters.
    :return: void
    """
    prob_infection = [0.02, 0.2, 0.8]
    seed_selection = ['rand', 'max_deg', 'min_deg']
    reinfection_factor = [0.5, 0.0]

    # Read the input graphs
    er = nx.read_edgelist('data/er-avg_deg-6.elist.txt')
    ws = nx.read_edgelist('data/ws-avg_deg-6.elist.txt')
    pc = nx.read_edgelist('data/pc-avg_deg-6.elist.txt')

    for p in prob_infection:
        for ss in seed_selection:
            for r in reinfection_factor:
                sir = SIRModel(prob_infection=p, length_of_infection=3,
                               reinfection_factor=r, num_seeds=2)

                er_sim = Simulator(er, sir, "er-p-{}-ss-{}-r-{}".format(p, ss, r), seed_selection=ss).run()
                ws_sim = Simulator(ws, sir, "ws-p-{}-ss-{}-r-{}".format(p, ss, r), seed_selection=ss).run()
                plc_sim = Simulator(pc, sir, "pc-p-{}-ss-{}-r-{}".format(p, ss, r), seed_selection=ss).run()

                contagion_stats_dict = {
                    'erdos-renyi': er_sim.contagion_stats,
                    'watts-strogatz': ws_sim.contagion_stats,
                    'powerlaw-cluster': plc_sim.contagion_stats
                }

                plot('SIRModel-p-{}-ss-{}-r-{}'.format(p, ss, r), contagion_stats_dict)

                avg_metrics = {
                    'erdos-renyi': er_sim.metrics,
                    'watts-strogatz': ws_sim.metrics,
                    'powerlaw-cluster': plc_sim.metrics
                }

                for i in range(10):
                    er_sim = Simulator(er, sir, "er-p-{}-ss-{}-r-{}".format(p, ss, r), seed_selection=ss).run()
                    ws_sim = Simulator(ws, sir, "ws-p-{}-ss-{}-r-{}".format(p, ss, r), seed_selection=ss).run()
                    plc_sim = Simulator(pc, sir, "pc-p-{}-ss-{}-r-{}".format(p, ss, r), seed_selection=ss).run()

                    for metric in er_sim.metrics.keys():
                        avg_metrics['erdos-renyi'][metric] = float((avg_metrics['erdos-renyi'][metric] + er_sim.metrics[metric])) / 2
                        avg_metrics['watts-strogatz'][metric] = float((avg_metrics['watts-strogatz'][metric] + ws_sim.metrics[metric])) / 2
                        avg_metrics['powerlaw-cluster'][metric] = float((avg_metrics['powerlaw-cluster'][metric] + plc_sim.metrics[metric])) / 2

                print "Average metrics for: p: {}, ss: {}, r: {}".format(p, ss, r)
                pprint(avg_metrics)


def main(argv):
    if argv == 'gen':
        generate_graphs()
    elif argv == 'experiment':
        experiment()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main('experiment')

import networkx as nx
from numpy.random import choice as rand
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod


class EpidemicModel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def init_graph(self, graph):
        raise NotImplementedError(
            'Function init_graph() should initialize the graph with attributes required by the model.')

    @abstractmethod
    def init_infection(self, graph):
        raise NotImplementedError('Function init_infection should infect some seed nodes.')

    @abstractmethod
    def apply(self, node, graph, visited, step):
        raise NotImplementedError('Function apply() should update the input node according to the model.')


num_figures = 0
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']  # colors to use in plots.


class Simulator:
    MAX_STEPS = 1000

    def __init__(self, g, model, name="graph"):
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
        self.infected = model.init_infection(g)
        self.contagion_stats = {self.steps_taken: self.infected}
        self.has_infection = True

    def step(self):
        self.steps_taken += 1
        new_infections = set()

        visited = set()
        self.has_infection = False

        for n in self.g.nodes():
            if n not in visited:
                visited, has_infection = self.model.apply(node=n, graph=self.g, visited=visited, step=self.steps_taken)
                self.has_infection |= has_infection
            if self.g.node[n]['state'][-1][0] == 1:
                new_infections.add(n)

        self.contagion_stats[self.steps_taken] = new_infections

    def run(self, steps_to_take=-1):
        if steps_to_take == -1:
            steps_to_take = self.MAX_STEPS

        for i in range(steps_to_take):
            self.step()
            if self.has_infection is False:
                print 'No more infections after step: {}'.format(self.steps_taken)
                break

        self.write_graph(0)

        return self.contagion_stats

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
        self.prob_infection = prob_infection
        self.length_of_infection = length_of_infection
        self.reinfection_factor = reinfection_factor
        self.num_seeds = num_seeds

    def init_graph(self, graph):
        for n in graph.nodes():
            state = list()
            state.append((self.SUSCEPTIBLE, 0.0, None))

            # print "Initial state: {}".format(state)
            graph.node[n][self.STATE] = state
            graph.node[n][self.PROB_INFECTION] = self.prob_infection
            graph.node[n][self.LENGTH_OF_INFECTION] = 0

        return graph

    def init_infection(self, graph):
        infected = set(rand(graph.nodes(), size=self.num_seeds))
        for n in infected:
            graph.node[n][self.STATE][-1] = (self.INFECTED, 0.0, None)
            print "Infecting: {}".format(n)
            print graph.node[n][self.STATE]

        return infected

    def apply(self, node, graph, visited, step=0):
        def update_visited_list(n):
            visited.add(n)
            attr = graph.node[n]
            if attr[self.STATE][-1][0] == self.INFECTED:
                return True
            else:
                return False

        def update_susceptible(n):
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


def draw(g, name="graph"):
    nx.draw(g)
    plt.savefig("{}.png".format(name), format="PNG")


def main():
    num_nodes = 100
    er = nx.erdos_renyi_graph(n=num_nodes, p=0.2)

    sir = SIRModel(prob_infection=0.02, length_of_infection=3, reinfection_factor=0.5, num_seeds=2)

    simulator = Simulator(er, sir, "er")
    simulator.run()

    contagion_stats_dict = {'erdos-renyi': simulator.contagion_stats}

    ws = nx.watts_strogatz_graph(n=num_nodes, k=5, p=0.2)
    contagion_stats_dict['watts-strogatz'] = Simulator(ws, sir, "ws").run()

    pc = nx.powerlaw_cluster_graph(num_nodes, m=5, p=0.2)
    contagion_stats_dict['powerlaw-cluster'] = Simulator(pc, sir, "pc").run()

    plot('SIRModel', contagion_stats_dict)


if __name__ == "__main__":
    main()

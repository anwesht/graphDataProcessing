from generator import Generator
from analyzer import Analyzer

import networkx as nx
import matplotlib.pyplot as plt
import sys

FACEBOOK = "./dataset/fb107.txt"
ARXIV = "./dataset/caGrQc.txt"
num_figures = 0  # the number of figures drawn.
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']  # colors to use in plots.


def draw(g):
    """
    Draw the graph
    :param g: graph to draw
    :type g: Graph
    """
    nx.draw(g, with_labels=True)
    # plt.show()


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
    degree_dict = map(lambda (k,v): (k, float(v)/g.number_of_nodes()), degree_dict)
    return degree_dict


def plot_degree_distribution(name, grouped_degree_dict):
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
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel("log (k)")  # Degree
    ax.set_ylabel("log (Pk)")  # frequency/ degree distribution

    grouped_degree_dict = sorted(grouped_degree_dict.items())

    i = 0
    for (key, degree_dict) in grouped_degree_dict:
        x_axis_vals = [k for (k, _) in degree_dict.items()]
        y_axis_vals = [v for (_, v) in degree_dict.items()]

        ax.plot(x_axis_vals, y_axis_vals, color=COLORS[i], label=key)

        i = (i + 1) % len(COLORS)

    ax.legend(loc='best')
    fig.savefig(name)


def plot_assortativity(name, m_dict):
    """
    Generate a graph of assortativity (r) vs probability of leader (p)
    :param name: name of the graph
    :type name: str
    :param m_dict: dictionary with key = m value, dictionary with key = p and value = assortativity
    :type m_dict: dict[int, dict[float, float]]
    """
    print "Plotting assortativity"

    global num_figures
    fig = plt.figure(num_figures)
    fig.suptitle('Plot: Assortativity coefficient (r) vs p\n for: {}'.format(name))

    num_figures += 1

    ax = fig.add_subplot(111)
    ax.set_xlabel("p")  # Probability of leader
    ax.set_ylabel("r")  # Assortativity

    i = 0
    for (m, p_dict) in m_dict.items():
        p_dict = sorted(p_dict.items(), key=lambda t: t[0])
        x_axis_vals = [k for (k, _) in p_dict]
        y_axis_vals = [v for (_, v) in p_dict]

        ax.plot(x_axis_vals, y_axis_vals, color=COLORS[i], label=m, marker='o')
        i = (i + 1) % len(COLORS)

    ax.legend(loc='best')
    fig.savefig(name)


def get_output_file(graph_name, n, p, m, l, i):
    """
    Generate output file name.
    :param graph_name: Name of the graph
    :param n: number of nodes
    :param p: probability of leader
    :param m: number of links added in each time step
    :param l: penetration depth
    :param i: iteration number
    :return: Name of the graph
    :rtype: str
    """
    return "{}-n{}-p{}-m{}-v{}-iter{}-edges.txt".format(graph_name, n, p, m, l, i)


def generate_graphs(graph_name, n0=10, n=100, l=1, m_list=[2, 5, 10], p_list=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
    """
    Generate a synthetic graph based on provided parameters.
    :param graph_name: Name of the graph
    :param n0: Number of nodes to start with. If len(m) > 1, using m as the value of n0.
    :param n: Total number of nodes in resulting graph
    :param l: Penetration depth
    :param m_list: list of values of m (number of links formed for each new node)
    :param p_list: list of values of p (probability of leader)
    :return: avg_degree_dict_by_p
    :rtype: dict[str, dict[int, float]]
    """
    assortativities = {}
    num_runs = 10
    graphs = []
    avg_degree_dict_by_p = {}

    for m in m_list:
        if len(m_list) > 1:
            n0 = m

        p_r_dict = {}
        for p in p_list:
            sum_of_r = 0.0
            avg_degree_dict = {}

            for i in range(0, num_runs):
                graph = Generator(n0=n0, n=n, l=l, m=m, p=p).generate()
                of = get_output_file(graph_name, n, p, m, l, i)
                nx.write_edgelist(graph, of, data=False)

                graphs.append(graph)

                degree_dict = dict(get_degree_dict(graph))
                if len(avg_degree_dict) == 0:
                    avg_degree_dict = degree_dict
                else:
                    for (k, v) in degree_dict.items():
                        if k in avg_degree_dict:
                            avg_degree_dict[k] = float((avg_degree_dict[k] + v))/2
                        else:
                            avg_degree_dict[k] = float(v)

                r = nx.degree_assortativity_coefficient(graph)
                sum_of_r += r
            # ----
            avg_degree_dict_by_p[str(p)] = avg_degree_dict
            r_avg = sum_of_r / num_runs
            p_r_dict[p] = r_avg

        degree_distr_name = "{}_n0={}_n={}_l={}_m={}_degree_dist.png".format(graph_name, n0, n, l, m)
        plot_degree_distribution(degree_distr_name, avg_degree_dict_by_p)

        assortativities[m] = p_r_dict

    print "Assortativities: {}".format(assortativities)
    assortativities_name = "{}-n={}-assortativity.png".format(graph_name, n)
    plot_assortativity(assortativities_name, assortativities)
    return avg_degree_dict_by_p


def calculate_local_assortativity(node, graph):
    alpha = 0.0
    for n in graph.neighbors(node):
        alpha += (graph.degree(n) - 1)

    alpha *= ((graph.degree(node) - 1)/(2 * graph.number_of_edges()))


def main(argv):
    if argv == "facebook":
        facebook = nx.read_edgelist(path=FACEBOOK, create_using=nx.Graph())
        Analyzer(facebook, "facebook").analyze()

        grouped_degree_dict = generate_graphs("facebook",
                                              n0=30,
                                              n=facebook.number_of_nodes(),
                                              l=1,
                                              m_list=[25],
                                              p_list=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Uncomment to generate the graph with selected parameters
        # grouped_degree_dict = generate_graphs("facebook", n0=30, n=facebook.number_of_nodes(), l=1, m_list=[25],
        #                                     p_list=[0.43])  #p =0.43

        grouped_degree_dict["facebook"] = dict(get_degree_dict(facebook))
        plot_degree_distribution("facebook", grouped_degree_dict)
    elif argv == "arxiv":
        arxiv = nx.read_edgelist(path=ARXIV, create_using=nx.Graph())
        Analyzer(arxiv, "arxiv").analyze()

        grouped_degree_dict = generate_graphs("arxiv",
                                              n0=5,
                                              n=arxiv.number_of_nodes(),
                                              l=1,
                                              m_list=[3],
                                              p_list=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Uncomment to generate the graph with selected parameters
        # grouped_degree_dict = generate_graphs("arxiv", n0=5, n=arxiv.number_of_nodes(), l=3, m_list=[3],
        #                                       p_list=[0.95])

        grouped_degree_dict["arxiv"] = dict(get_degree_dict(arxiv))
        plot_degree_distribution("arxiv", grouped_degree_dict)
    elif argv == "synth":
        for n in [1000, 5000, 10000]:
            generate_graphs("synth", n=n)
    else:
        print "usage: python main.py facebook | arxiv | synth"
        sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
    plt.show()


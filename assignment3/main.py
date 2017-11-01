from generator import Generator
from analyzer import Analyzer
import networkx as nx
import matplotlib.pyplot as plt


FACEBOOK = "./dataset/fb107.txt"
ARXIV = "./dataset/caGrQc.txt"
OUTPUT_DIR = "SYNTHGRAPHS/"


def draw(g):
    print g.number_of_nodes()
    nx.draw(g, with_labels=True)
    # plt.show()


def get_degree_dict(g):
    degree_dict = {}
    for n in g.nodes():
        d = g.degree(n)
        if d not in degree_dict:
            degree_dict[d] = 0
        degree_dict[d] += 1
    degree_dict = sorted(degree_dict.items())
    degree_dict = map(lambda (k,v): (k, float(v)/g.number_of_nodes()), degree_dict)
    return degree_dict


def plot_degree_distribution(g1, *graphs):
    """
    Counts the frequency of each degree and plots it in a log-log plot.
    :param g: graph for which the degree distribution is to be plotted.
    :param graph_file_name: current edgelist name
    :return: void
    """
    colors = ['r', 'b', 'g']

    degree_dict_list = [get_degree_dict(g1)]

    for g in graphs:
        degree_dict_list.append(get_degree_dict(g))

    plt.title("Plot of the degree distribution")
    plt.xlabel("log (k)")  # Degree
    plt.ylabel("log (Pk)") # frequency/ degree distribution

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    i = 0
    for degree_dict in degree_dict_list:
        x_axis_vals = [k for (k, _) in degree_dict]
        y_axis_vals = [v for (_, v) in degree_dict]

        ax.plot(x_axis_vals, y_axis_vals, color=colors[i])
        i += 1

    # Uncomment to view plt
    # plt.show()
    # fig.savefig(self.graph_name+ "-degree_distribution.png")


def plot_assortativity(m_dict):
    colors = ['r', 'b', 'g', 'b']

    print "Plotting assortativity"

    plt.title("Plot: Assortativity coefficient (r) vs p")
    plt.xlabel("p")  # Probability of leader
    plt.ylabel("r")  # Assortativity

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.legend(loc="bottom right")

    i = 0
    for (m, p_dict) in m_dict.items():
        x_axis_vals = [k for (k, _) in p_dict]
        y_axis_vals = [v for (_, v) in p_dict]

        ax.plot(x_axis_vals, y_axis_vals, color=colors[i], label=m)
        i = (i + 1) % 4


def get_output_file(graph_name, n, p, m, l):
    # return "{}{}/{}-n{}-p{}-m{}-v{}-edges.txt".format(OUTPUT_DIR, graph_name, graph_name, n, p, m, l)
    return "{}-n{}-p{}-m{}-v{}-edges.txt".format(graph_name, n, p, m, l)


def generate_graphs(graph_name, n, l=1):
    m_list = [2, 5, 10]
    p_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    assortativities = {}
    for m in m_list:
        for p in p_list:
            sum_of_r = 0.0
            r_p_dict = {}
            for i in range(0, 10):
                graph = Generator(n0=m+1, n=n, l=1, m=m, p=p).generate()
                of = get_output_file(graph_name, n, p, m, l)
                nx.write_edgelist(graph, of, data=False)

                sum_of_r += nx.degree_assortativity_coefficient(graph)
            r = sum_of_r/10
            r_p_dict[r] = p
        assortativities[m] = r_p_dict

    plot_assortativity(assortativities)


def main():
    generator = Generator(3, 10, 1, 2, 1.0)
    graph = generator.generate()
    analyzer = Analyzer(graph, "generated")
    analyzer.analyze()
    draw(graph)

    facebook = nx.read_edgelist(path=FACEBOOK, create_using=nx.Graph())
    Analyzer(facebook, "facebook").analyze()

    arxiv = nx.read_edgelist(path=ARXIV, create_using=nx.Graph())
    Analyzer(arxiv, "arxiv").analyze()

    analyzer.plot_degree_distribution(facebook)

    # generate_graphs(FACEBOOK.split('/')[-1].split('.')[0], facebook.number_of_nodes())
    generate_graphs("fb107", facebook.number_of_nodes())


if __name__ == "__main__":
    main()
    plt.show()
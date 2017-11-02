from generator import Generator
from analyzer import Analyzer
import networkx as nx
import matplotlib.pyplot as plt


FACEBOOK = "./dataset/fb107.txt"
ARXIV = "./dataset/caGrQc.txt"
OUTPUT_DIR = "SYNTHGRAPHS/"
num_figures = 0
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


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


# def plot_degree_distribution(g1, *graphs):
def plot_degree_distribution(g1, graphs):
    """
    Counts the frequency of each degree and plots it in a log-log plot.
    :param g1: graph for which the degree distribution is to be plotted.
    :param graph_file_name: current edgelist name
    :return: void
    """
    global num_figures

    degree_dict_list = [get_degree_dict(g1)]

    for g in graphs:
        degree_dict_list.append(get_degree_dict(g))

    # plt.title("Plot of the degree distribution")
    plt.xlabel("log (k)")   # Degree
    plt.ylabel("log (Pk)")  # frequency/ degree distribution

    fig = plt.figure(num_figures)
    num_figures += 1

    fig.suptitle("Plot of the degree distribution")

    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    i = 0
    for degree_dict in degree_dict_list:
        x_axis_vals = [k for (k, _) in degree_dict]
        y_axis_vals = [v for (_, v) in degree_dict]

        ax.plot(x_axis_vals, y_axis_vals, color=COLORS[i])
        # i += 1
        i = (i + 1) % len(COLORS)

    # fig.show()
    # Uncomment to view plt
    # plt.show()
    fig.savefig("degree_distribution_facebook.png")


def plot_degree_distribution_from_dict(name, degree_dict_by_p):
    # plt.title("Plot of the degree distribution \nfor: {}".format(name))
    plt.xlabel("log (k)")   # Degree
    plt.ylabel("log (Pk)")  # frequency/ degree distribution

    global num_figures

    fig = plt.figure(num_figures)
    num_figures += 1
    fig.suptitle("Plot of the degree distribution \nfor: {}".format(name))

    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    degree_dict_by_p = sorted(degree_dict_by_p.items())

    i = 0
    for (p, degree_dict) in degree_dict_by_p:
        x_axis_vals = [k for (k, _) in degree_dict.items()]
        y_axis_vals = [v for (_, v) in degree_dict.items()]

        ax.plot(x_axis_vals, y_axis_vals, color=COLORS[i], label=p)

        i = (i + 1) % len(COLORS)

    ax.legend(loc='best')
    fig.savefig(name)


def plot_assortativity(name, m_dict):
    print "Plotting assortativity"

    # plt.title("Plot: Assortativity coefficient (r) vs p")
    plt.xlabel("p")  # Probability of leader
    plt.ylabel("r")  # Assortativity

    global num_figures
    fig = plt.figure(num_figures)
    fig.suptitle('Plot: Assortativity coefficient (r) vs p\n for: {}'.format(name))

    num_figures += 1

    ax = fig.add_subplot(111)

    # m_dict = sorted(m_dict.items(), key=lambda t: t[0])

    # print "m_dict: {}".format(m_dict)
    i = 0
    for (m, p_dict) in m_dict.items():
        p_dict = sorted(p_dict.items(), key=lambda t: t[0])
        # print "pdict: {}".format(p_dict)
        x_axis_vals = [k for (k, _) in p_dict]
        y_axis_vals = [v for (_, v) in p_dict]

        ax.plot(x_axis_vals, y_axis_vals, color=COLORS[i], label=m, marker='o')
        i = (i + 1) % len(COLORS)

    ax.legend(loc='best')
    fig.savefig(name)


def get_output_file(graph_name, n, p, m, l, i):
    # return "{}{}/{}-n{}-p{}-m{}-v{}-edges.txt".format(OUTPUT_DIR, graph_name, graph_name, n, p, m, l)
    return "{}-n{}-p{}-m{}-v{}-iter{}-edges.txt".format(graph_name, n, p, m, l, i)


def generate_graphs(graph_name, n0=10, n=100, l=1, m_list=[2, 5, 10], p_list=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
    # m_list = [5]
    # p_list = [0.0, 0.2, 0.4, 0.6]
    # m_list = [2, 5, 10]
    # p_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    assortativities = {}

    num_runs = 10
    graphs = []
    avg_degree_dict_by_p = {}
    for m in m_list:
        n0 = m
        p_r_dict = {}
        for p in p_list:
            sum_of_r = 0.0
            avg_degree_dict = {}

            for i in range(0, num_runs):
                graph = Generator(n0=n0, n=n, l=l, m=m, p=p).generate()
                # graph = Generator(m, n, 1, m, p).generate()
                of = get_output_file(graph_name, n, p, m, l, i)
                nx.write_edgelist(graph, of, data=False)

                # draw(graph)

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
            avg_degree_dict_by_p[p] = avg_degree_dict
            r_avg = sum_of_r / num_runs
            p_r_dict[p] = r_avg

        degree_distr_name = "{}_n0={}_n={}_l={}_m={}_degree_dist.png".format(graph_name, n0, n, l, m)
        plot_degree_distribution_from_dict(degree_distr_name, avg_degree_dict_by_p)

        assortativities[m] = p_r_dict

    # plot_degree_distribution(graphs.pop(0), graphs)
    # plot_degree_distribution_from_dict(avg_degree_dict)
    print "Assortativities: {}".format(assortativities)
    assortativities_name = "{}-n={}-assortativity.png".format(graph_name, n)
    plot_assortativity(assortativities_name, assortativities)


def main():
    generator = Generator(3, 50, 1, 2, 0.6)
    graph = generator.generate()
    analyzer = Analyzer(graph, "generated")
    analyzer.analyze()
    # draw(graph)

    facebook = nx.read_edgelist(path=FACEBOOK, create_using=nx.Graph())
    Analyzer(facebook, "facebook").analyze()

    # draw(facebook)
    # plot_degree_distribution(facebook, [])

    arxiv = nx.read_edgelist(path=ARXIV, create_using=nx.Graph())
    Analyzer(arxiv, "arxiv").analyze()

    plot_degree_distribution(arxiv, [])
    # analyzer.plot_degree_distribution(facebook)

    # generate_graphs(FACEBOOK.split('/')[-1].split('.')[0], facebook.number_of_nodes())
    # generate_graphs("fb107", facebook.number_of_nodes())
    # generate_graphs("synth", n=10000)
    # generate_graphs("fb107", n0=25, n=facebook.number_of_nodes(), l=1, m_list=[25], p_list=[0.65])
    # generate_graphs("fb107", n0=25, n=facebook.number_of_nodes(), l=1, m_list=[25], p_list=[0.75])
    # generate_graphs("fb107", n0=25, n=facebook.number_of_nodes(), l=1, m_list=[25])
    # generate_graphs("fb107", n0=25, n=facebook.number_of_nodes(), l=1, m_list=[25], p_list=[0.5])
    # generate_graphs("fb107", n0=30, n=facebook.number_of_nodes(), l=3, m_list=[25], p_list=[0.43])  # saved this.
    # generate_graphs("arxiv", n0=5, n=arxiv.number_of_nodes(), l=1, m_list=[3], p_list=[0.9])
    for n in [1000, 500]:
        generate_graphs("synth", n=n)


if __name__ == "__main__":
    main()
    plt.show()
    # raw_input()

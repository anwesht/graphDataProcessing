from generator import Generator
from analyzer import Analyzer
import networkx as nx
import matplotlib.pyplot as plt


FACEBOOK = "./dataset/fb107.txt"
ARXIV = "./dataset/caGrQc.txt"
OUTPUT_DIR = "./SYNTHGRAPHS/"

def draw(g):
    print g.number_of_nodes()
    nx.draw(g, with_labels=True)
    # plt.show()


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


if __name__ == "__main__":
    main()
    plt.show()
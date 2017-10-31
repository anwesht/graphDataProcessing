from generator import Generator
import networkx as nx
import matplotlib.pyplot as plt


def draw(g):
    print g.number_of_nodes()
    nx.draw(g, with_labels=True)
    plt.show()


def main():
    generator = Generator(3, 10, 1, 2, 1.0)
    graph = generator.generate()
    draw(graph)


if __name__ == "__main__":
    main()
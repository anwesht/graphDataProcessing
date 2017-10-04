import sys
import networkx as nx
import itertools
import matplotlib.pyplot as plt
from random import random


def main(argv):
    if len(argv) != 1:
        print "usage: python compare-food.py <path/to/allr_recipes.txt>"
        sys.exit(0)

    file_path = argv[0]

    # Change the criteria for the graphs here.
    mix_criteria = ['Canada', 'Scandinavia']

    recipes_list = []
    count = 0

    with open(file_path, 'r') as f:
        for line in f:
            split_line = line.split()
            country = split_line[0]

            if country in mix_criteria:
                ingredients = split_line[1:]
                recipes_list.append(ingredients)
                count += 1

    g_nx = nx.Graph()

    for recipe in recipes_list:
        edges = itertools.combinations(recipe, 2)
        g_nx.add_edges_from(edges)

    degrees = nx.degree(g_nx)

    filtered_nodes = [n for n in degrees if degrees[n] > 100]
    sub_graph = g_nx.subgraph(filtered_nodes)

    ds = nx.degree(sub_graph)

    plt.figure(figsize=(14, 14))

    nx.draw_random(sub_graph, with_labels=True,
                   nodelist=ds.keys(), node_size=[d * 10 for d in ds.values()],
                   font_size=8, edge_color='grey')
    # nx.draw_random(g_nx, with_labels=True,
    #                nodelist=degrees.keys(), node_size=[d * 10 for d in degrees.values()],
    #                font_size=8, edge_color='grey')

    # plt.show()

    output_file = "-".join(mix_criteria) + ".elist.txt"
    # plt.savefig(output_file + "-subgraph.png")

    print "Edge list is in: {}".format(output_file)

    # nx.write_edgelist(g_nx, output_file, data=False)


if __name__ == "__main__":
    main(sys.argv[1:])
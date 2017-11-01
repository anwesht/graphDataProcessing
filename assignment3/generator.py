import networkx as nx
import matplotlib.pyplot as plt
from numpy.random import choice as rand
from itertools import groupby


class Generator:
    FOLLOWER = 1
    LEADER = 2

    def __init__(self, n0, n, l, m, p):
        self.num_start_nodes = n0
        self.num_nodes = n
        self.penetration_depth = l
        self.num_links = m
        self.prob_leader = p
        self.prob_follower = 1 - p
        self.graph = nx.complete_graph(n0)

    def __str__(self):
        return "Graph parameters: \n" \
               + "Number of nodes in initial network (N0): {} \n".format(self.num_start_nodes) \
               + "Required number of nodes (N): {} \n".format(self.num_nodes) \
               + "Penetration depth(l): {} \n".format(self.penetration_depth) \
               + "Number of links formed in each time step (m): {}\n".format(self.num_links) \
               + "Probablity that a new node is a follower (type I) (1 - p): {} \n".format(self.prob_follower) \
               + "Probablity that a new node is a leader (type II) (p): {}".format(self.prob_leader)

    def get_type_of_node(self):
        return rand([self.FOLLOWER, self.LEADER], p=[self.prob_follower, self.prob_leader])

    @staticmethod
    def get_random_node(node_list):
        rand_index = rand(range(0, len(node_list)))
        return node_list.pop(rand_index)[0]

    def add_follower_edge(self, current_node, g):
        self.graph.add_node(current_node, attr_dict={'type': self.FOLLOWER})
        nodes = list(g.nodes())
        for _ in range(0, self.num_links):
            if len(nodes) == 0:
                break
            # n = rand(g.nodes())
            n = self.get_random_node(nodes)
            self.graph.add_edge(current_node, n)

    def add_leader_edge(self, current_node, g):
        self.graph.add_node(current_node, attr_dict={'type': self.LEADER})

        sorted_degrees = sorted(g.degree(), key=lambda t: t[1])

        group = list(groupby(sorted_degrees, key=lambda t: t[1]))
        group_index = 0

        current_group = []

        # Randomly selecting a node with minimum degree instead of simply popping from sorted_degrees
        for _ in range(0, self.num_links):
            if len(current_group) > 0:
                n = self.get_random_node(current_group)
            elif group_index < len(group):
                print "-------------------------"
                print "group {}: {}".format(group_index, group)
                print "group index: {}".format(group[group_index])
                current_group = list(group[group_index][1])
                print "current group {}".format(current_group)
                group_index += 1
                if len(current_group) == 0:
                    break
                n = self.get_random_node(current_group)
            else:
                break
            # print ("adding edge to node {} with degree {}".format(n, g.degree(n)))
            self.graph.add_edge(current_node, n)

            # Simply picking the next node in sorted_degrees.
            # for _ in range(0, self.num_links):
            #     n = sorted_degrees.pop(0)[0]
            #     self.graph.add_edge(current_node, n)

    def generate(self):
        for t in range(self.num_start_nodes, self.num_nodes):
            # print ("Time step: {}".format(t))

            # Choose a random node as anchor node.
            anchor_node = rand(self.graph.nodes())

            # Generate sub-graph (Gj) of depth penetration_depth
            sub_graph = nx.ego_graph(self.graph, anchor_node, self.penetration_depth)

            # Set type of the node to add
            type_of_node = self.get_type_of_node()

            if type_of_node == self.FOLLOWER:
                self.add_follower_edge(t, sub_graph)
            else:
                self.add_leader_edge(t, sub_graph)

        return self.graph

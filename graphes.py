import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# python-igraph manual
# https://igraph.org/python/tutorial/latest/generation.html

def adjacency_matrix(n : int, prob : float):
    """
    generate an adjency matrix n x n
    """
    # return np.random.randint(2, size=(n,n))
    return np.random.choice([0,1], size=(n,n), replace=True, p=[1-prob, prob])

def main():

    number_of_vertices = 6
    prob_connexion_between_vertices = 0.3

    A = adjacency_matrix(number_of_vertices, prob_connexion_between_vertices)
    print(A)
    G = nx.from_numpy_array(A)
    # nx.draw(G, with_labels=1)

    G1=nx.MultiDiGraph()
    G1.add_edge('A', 'B', weight=6, relation='famille')
    G1.add_edge('A', 'B', weight=18, relation='ami')
    G1.add_edge('B', 'C', weight=13, relation='ami')
    G1.add_node('A', role='trader')
    G1.add_node('B', role='trader')
    G1.add_node('C', role='manager')
    # nx.draw(G1, with_labels=1)

    print(list(G1.edges(data=True)))
    print(list(G1.nodes(data=True)))
    print(G1['A']['B'])
    print(G1['A']['B'][0]['weight'])
    print(G1.nodes['A']['role'])

    from networkx import bipartite
    B = nx.Graph()
    B.add_nodes_from(['A', 'B', 'C', 'D', 'E'], bipartite=0)
    B.add_nodes_from([1, 2, 3, 4], bipartite=1)
    B.add_edges_from([('A', 1), ('B', 1), ('C', 1), ('C', 3), ('D', 2), ('D', 3), ('E', 3), ('E', 4)])
    # getting each set of nodes of a bipartite graph
    print(bipartite.sets(B))
    print(bipartite.is_bipartite(B))
    B.add_edge('A', 'B')
    print(bipartite.is_bipartite(B))
    B.remove_edge('A', 'B')
    X = set([1, 2, 3, 4])
    print(bipartite.is_bipartite_node_set(B, X))
    X = set(['A','B','C','D','E'])
    print(bipartite.is_bipartite_node_set(B, X))

    B=nx.Graph()
    B.add_edges_from([('A',1), ('B',1), ('C',1), ('D',1), ('H',1),
                      ('B',2), ('C',2), ('D',2), ('E',2), ('G',2),
                      ('E',3), ('F',3), ('H',3), ('J',3), ('E',4),
                      ('I',4), ('J',4)])
    X=set(['A','B','C','D','E','F','G','H','I','J'])
    P=bipartite.projected_graph(B,X)
    X=set([1,2,3,4])
    P=bipartite.weighted_projected_graph(B,X)
    nx.draw(P, with_labels=1)
    plt.show()


# for i in range(1000):
# #creates one number out of 0 or 1 with prob p 0.4 for 0 and 0.6 for 1
#     test = numpy.random.choice(numpy.arange(0, 2), p=[0.4, 0.6])
#     myProb.append(test)


if __name__ == '__main__':
    main()
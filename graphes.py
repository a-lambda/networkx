import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# python-igraph manual
# https://igraph.org/python/tutorial/latest/generation.html

number_of_vertices = 6
prob_connexion_between_vertices = 0.3

def adjacency_matrix(n : int, prob : float):
    """
    generate an adjency matrix n x n
    """
    # return np.random.randint(2, size=(n,n))
    return np.random.choice([0,1], size=(n,n), replace=True, p=[1-prob, prob])

def main():
    A = adjacency_matrix(number_of_vertices, prob_connexion_between_vertices)
    print(A)
    G = nx.from_numpy_matrix(A)
    nx.draw(G, with_labels=1)
    plt.show()

# for i in range(1000):
# #creates one number out of 0 or 1 with prob p 0.4 for 0 and 0.6 for 1
#     test = numpy.random.choice(numpy.arange(0, 2), p=[0.4, 0.6])
#     myProb.append(test)


if __name__ == '__main__':
    main()
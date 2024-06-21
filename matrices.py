import numpy as np

np.set_printoptions(suppress=True) # permet d'occulter les problèmes d'arrondi

A = np.array([
                [1,2,3], 
                [4,5,6],
            ])
B = np.array([
                [7,8,9],
                [10,11,12],
            ])
C = np.add(A, B)
D = np.multiply(A, B)
E = np.subtract(A, B)
F = np.divide(A, B)
G = np.sum(A)
H = np.sum(A, axis=0)
I = np.sum(A, axis=1)
J = A.T

print(C)
print(D)
print(E)
print(F)
print(G)
print(H)
print(I)
print(J)

A1 = np.array([
                [1,3,-4], 
                [-3,-2,1],
                [4,-1,-1]
             ])

K = np.dot(A1.T, A1)
print(K)
L = np.diag(A1)
M = np.diag(A1,k=1)
N = np.diag(A1,k=-1)
O = np.trace(A1)
print(L)
print(M)
print(N)
print(O)
P = np.triu(A1)
Q = np.tril(A1)
print(P)
print(Q)
R = np.dot(A1, A1)
print(R)
det = np.linalg.det(A1)
print(det)
print(type(A1[1,1]))
print(type(det))

A = np.array([[1,2],[3,4]],dtype=int)
B = np.linalg.inv(A)
C = np.dot(A,B)
D = C.astype(int)
print(A)
print(B)
print(C)
print(D)
S = np.linalg.matrix_rank(A)
print(S)

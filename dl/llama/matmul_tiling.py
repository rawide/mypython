import numpy as np

A = np.random.randint(1, 10, (8,4))
B = np.random.randint(1, 10, (4,4))
AB = np.dot(A, B)
# print(AB[6][3])
C = np.zeros(32).reshape(8,4)
# print(C)
step = 2
for i in range(0, 8, step):
    for j in range(0, 4, step):
        for k in range(0, 4, step):
            for s in range(0, step):
                C[i][j]     += A[i][k+s]*B[k+s][j]
                C[i][j+1]   += A[i][k+s]*B[k+s][j+1]
                C[i+1][j]   += A[i+1][k+s]*B[k+s][j]
                C[i+1][j+1] += A[i+1][k+s]*B[k+s][j+1]

print(AB)
print(C)
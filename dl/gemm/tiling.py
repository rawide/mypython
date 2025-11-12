import numpy as np

A = np.random.randint(1, 10, (6,8))
B = np.random.randint(1, 10, (8,9))
AB = np.dot(A, B)
print("A=",A)
print("B=",B)
C = np.zeros(54).reshape(6,9)
# print(C)
step1 = 3
step2 = 2
step3 = 3
for i in range(0, 6, step1):
    for j in range(0, 9, step3):
        for k in range(0, 8, step2):
            for s in range(0, step2):
                C[i][j]     += A[i][k+s]*B[k+s][j]
                C[i][j+1]   += A[i][k+s]*B[k+s][j+1]
                C[i][j+2]   += A[i][k+s]*B[k+s][j+2]
                C[i+1][j]   += A[i+1][k+s]*B[k+s][j]
                C[i+1][j+1] += A[i+1][k+s]*B[k+s][j+1]
                C[i+1][j+2] += A[i+1][k+s]*B[k+s][j+2]
                C[i+2][j]   += A[i+2][k+s]*B[k+s][j]
                C[i+2][j+1] += A[i+2][k+s]*B[k+s][j+1]
                C[i+2][j+2] += A[i+2][k+s]*B[k+s][j+2]

print("AB=",AB)
print(C)

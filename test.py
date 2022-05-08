import numpy as np


new_rows = new_cols = 6
rows = cols = 5
# A = np.zeros((3, 2))
# print(A)
# B = np.zeros(3)
# print(B)

A = np.zeros((rows, cols))

for i in range(0,5):
    for j in range(0,5):
        A[i][j] = i+j
print(A)

B = np.zeros((new_rows, new_cols))
print(B)

B[0:rows][:,0:cols] = A

C = B[0:rows,0:cols]

print(B)
print('c:',C)
# coding: utf-8
a = [[0, 1], [2, 3]]
a[0,1]
a[0][1]

import numpy as np
cube = np.zeros((3, 3, 3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            cube[i,j,0] = 1
            cube[i,j,1] = 2
            cube[i,j,2] = 3
cube
for i in range(3):
    for j in range(3):
        for k in range(3):
            cube[i,j,0] = i
            cube[i,j,1] = j
            cube[i,j,2] = k
cube

cube[:,:,0] = 1
cube[:,:,1] = 2
cube[:,:,2] = 3
cube

cube[0,:,:] = 1
cube[1,:,:] = 2
cube[2,:,:] = 3
cube

flattened = np.reshape(cube, (9, 3))
flattened
kernel = np.array([
    [-10, 0, 1],
    [0.5, 0, 9],
    [1, -1, 2]
])
result = np.dot(flattened, kernel.T)
unflattened = np.reshape(result, (3, 3, 3))
unflattened

first = np.dot([[1, 1, 1], [1, 1, 1], [1, 1, 1]], kernel.T)
first
second = np.dot([[2, 2, 2], [2, 2, 2], [2, 2, 2]], kernel.T)
second
third = np.dot([[3, 3, 3], [3, 3, 3], [3, 3, 3]], kernel.T)
third

tensor = np.zeros((4, 2, 3))

for i in range(4):
    for j in range(2):
        for k in range(3):
            tensor[i,j,k] = 1 + i + (i*j) + (i*j*k)

flattened = np.reshape(tensor, (4*2, 3))
result = np.dot(flattened, kernel.T)
unflattened = np.reshape(result, (4, 2, 3))

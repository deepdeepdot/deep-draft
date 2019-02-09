# coding: utf-8

cube[0,:,:] = 1
cube[1,:,:] = 2
cube[2,:,:] = 3
import numpy as np
cube = np.zeros((3, 3, 3))

cube[0,:,:] = 1
cube[1,:,:] = 2
cube[2,:,:] = 3
cube
cube[:,:,0] = 1
cube[:,:,1] = 2
cube[:,:,2] = 3
cub
cube
cube[0,0,0]
cube[1,2,3]
cube[0,1,2]
cube[0,1,1]
cube[0,1,0]
cube[1,1,0]
cube[1,2,0]
cube[3,2,0]
cube[0,2,0]
cube[0,2,1]
cube[1,2,1]
cube[2,2,1]
cube[2,2,2]
flattened = np.reshape(cube, (9, 3))

kernel = np.array([
    [-1, 0, 1],
    [0, 1, 0],
    [2, 1, 2]
])
result = np.dot(flattened, kernel.T)
unflattened = np.reshape(result, (3, 3, 3))
result
unflattened
kernel = np.array([
    [-10, 0, 1],
    [0.5, 1, 0],
    [1, 1, 2]
])
result = np.dot(flattened, kernel.T)
result
result = np.dot(flattened, kernel)
result

kernel = np.array([
    [-10, 0, 1],
    [0.5, 0, 9],
    [1, -1, 2]
])
kernel = np.array([
    [-10, 0, 1],
    [0.5, 0, 9],
    [1, -1, 2]
])
result = np.dot(flattened, kernel.T)
result
flattened
input = np.random.random((3, 3, 3))
input
input = np.random.random((10, 50, 3))
input
input = np.random.random((4, 10, 3))
input
result = np.dot(input, kernel.T)
unflattened = np.reshape(result, (3, 3, 3))
flattened = np.reshape(input, (4*10, 3))
result = np.dot(flattened, kernel.T)
result
unflattened = np.reshape(flattened, (4, 10, 3))
unflattened
input[0,:,:]
input
first = np.dot(input[0,:,:], kernel.T)
first
first = np.dot(input[:,:,0], kernel.T)
first = np.dot(input[:,:0], kernel.T)
first
tensor = np.zeros((4, 2, 3))
for i in range(4):
    for j in range(2):
        for k in range(3):
            tensor[i,j,k] = i + (i*j) + (i*j*k)
tensor
for i in range(4):
    for j in range(2):
        for k in range(3):
            tensor[i,j,k] = 1 + i + (i*j) + (i*j*k)
tensor
flattened = np.reshape(tensor, (4*2, 3))
result = np.dot(flattened, kernel.T)
unflattened = np.reshape(result, (4, 2, 3))
flattened
result
unflattened
tensor


tensor[0,:,:]
first = np.dot(tensor[0,:,:], kernel.T)
first
second = np.dot(tensor[1,:,:], kernel.T)
second
get_ipython().run_line_magic('save', 'kernel_testing_2.py')

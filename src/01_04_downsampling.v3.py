import matplotlib.pyplot as plt
import numpy as np

s = stride = 2 # Try 6?
img = plt.imread("panda-corner.jpg")
nrows, ncols = img.shape[0], img.shape[1]
nchannels = img.shape[2]

half_rows, half_cols = int(nrows / stride), int(ncols / stride)

buffer = np.zeros((half_rows, half_cols, nchannels), dtype=int)

for row in range(stride, half_rows-stride):
    for col in range(stride, half_cols-stride):
        sub = img[row*s:(row+1)*s, col*s:(col+1)*s,:]
        for c in range(nchannels):
            buffer[row, col, c] = np.max(sub[:,:,c])

plt.imsave("out/panda-on-a-diet.v3.png", buffer)

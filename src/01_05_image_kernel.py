import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("panda-corner.jpg")
nrows, ncols = img.shape[0], img.shape[1]
nchannels = img.shape[2]

kernels = {
    'outline': [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ],
    'sharpen': [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ],
    'right_sobel': [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ],
    'blur': [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625]
    ]
}

for label, kernel in kernels.items():
    buffer = np.zeros((nrows, ncols, 3))
    # buffer = [[[0 for i in range(3)] for j in range(ncols)] for k in range(nrows)]

    for i in range(nrows):
        for j in range(ncols):
            if (i > 0 and j > 0) and (i < nrows-1 and j < ncols-1):
                for c in range(nchannels):
                    source = img[i-1:i+2, j-1:j+2, c]
                    buffer[i][j][c] = np.sum(np.multiply(source, kernel))

    buffer = np.clip(buffer, 0, 255).astype(int)
    plt.imsave(f"out/panda-{label}.png", buffer)

print("Done!")
## Where's your function?

![The Equation Image](../img/e=mc2.png)


### Math
- Function, polynomial: linear, quadratic, cubic (smooth fns)
- Plotting functions using matplotlib
- Find (maxima and) minima of a quadratic function

- How do we approximate a function?
- Slope = rate of change (ladder analogy, step 1, step2, step3??? step4?????)


### Numpy

    import numpy as np

    A = [[1, 2, 3, 8],
        [2, 0, 3, 9],
        [0, 1, 3, 1]]

    B = [[3, 1], [4, 4], [6, 5], [2, 0]]
    
    X = np.matmul(A, B)
    X.T

    [1, 2, 3] * 2
    np.array([2, 3, 3]) * 2 + 5

    [1, 2, 3] + [2, 3, 4]
    np.array([1, 2, 3]) + np.array([2, 3, 4])

    # array slicing
    B = np.zeros((3, 4, 5))
    C = B[:,:,]

    C[0, 0, 0] # what about array?

    A = [n for n in range(9)]
    A_2D = np.reshape(A, (3, 3))
    A_flat = np.reshape(A_2D, (9))
    A_2D.T

    B = np.array([n for n in range(27)])
    B_3D = B.reshape((3, 3, 3))
    B_flat = B_3D.reshape((3 * 3 * 3))
    B_3D.T


#### Ex: Downsampling
    import matplotlib.pyplot as plt
    import numpy as np

    s = stride = 2 # Try 6?
    img = plt.imread("panda-corner.jpg")
    nrows, ncols, nchannels = img.shape[0], img.shape[1], img.shape[2]
    half_rows, half_cols = int(nrows / stride), int(ncols / stride)

    buffer = np.zeros((half_rows, half_cols, nchannels), dtype=int)

    for row in range(stride, half_rows-stride):
        for col in range(stride, half_cols-stride):
            sub = img[row*s:(row+1)*s, col*s:(col+1)*s,:]
            for c in range(nchannels):
                buffer[row, col, c] = np.max(sub[:,:,c])


#### Ex: Image kernel

    buffer = np.zeros((nrows, ncols, 3))

    for i in range(nrows):
        for j in range(ncols):
            if (i > 1 and j > 1) and (i < nrows-1 and j < ncols-1):
                for c in range(nchannels):
                    source = img[i-1:i+2, j-1:j+2, c]
                    buffer[i][j][c] = np.sum(np.multiply(source, kernel))

    buffer = np.clip(buffer, 0, 255).astype(int)


### Jupyter Notebooks

- Run locally
        $ conda activate nanohackers
        $ jupyter notebook .

- Post a notebook
    - https://nbviewer.jupyter.org/
    - Share on your github as an .ipynb file

- Collaborate on a notebook (just google docs)
    - https://colab.research.google.com/



### Magenta.js

    Hello Magenta
    https://medium.com/@oluwafunmi.ojo/getting-started-with-magenta-js-e7ffbcb64c21
    https://hello-magenta.glitch.me/

    Melody Mixer
    https://experiments.withgoogle.com/ai/melody-mixer/view/
    https://github.com/googlecreativelab/melody-mixer



### Reference
Colab: https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c

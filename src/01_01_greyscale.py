import matplotlib.pyplot as plt

img = plt.imread("panda-corner.jpg")
nrows, ncols = img.shape[0], img.shape[1]

copy = img.copy()

for row in range(nrows):
    for col in range(ncols):
        avg = sum(img[row,col,:]) / 3
        copy[row,col,:] = avg

plt.imshow(copy), plt.show()
plt.imsave("out/panda-grey.png", copy)

# We can try the following
# nchannels = dim[2]
# copy = np.zeros(shape=(nrows, ncols, nchannels), dtype=int)

# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
img = mpimg.imread("panda-corner.jpeg")
img = mpimg.imread("panda-corner.jpg")
img.size
img.shape
img
copy = img.clone()
copy = img.copy()
copy
img[0,0,0]
img[0,0,0] = 127
copy[0,0,0]
copy[0,0,0] = 127
copy.shape
copy.shape[0]
copy.shape[1]
copy.shape[2]
dim = copy.shape
for row in range(dim[0]):
    for col in range(dim[1]):
        avg = np.sum(img[row,col,:])
        copy[row,col,:] = avg
        
        
mpimg.save("grey.png", copy)
mplt.image.imsave("grey.png", copy)
plt.image.imsave("grey.png", copy)
plt.imsave("grey.png", copy)
get_ipython().run_line_magic('s', 'greyscale-panda.py 1-25')

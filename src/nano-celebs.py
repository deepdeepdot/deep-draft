import matplotlib.pyplot as plt
import numpy as np
import os

files = os.listdir("celebrity_images")
files

files[50]
img = plt.imread(os.path.join("celebrity_images", files[50]))
img.shape
plt.imshow(img)
plt.show()

img[0,0,0]
img[0,0,1]
img[0,0,2]
plt.imshow(img[:,:,0], cmap='gray')
plt.imshow(img[:,:,1], cmap='gray')
plt.imshow(img[:,:,2], cmap='gray')


paths = [os.path.join("celebrity_images/", f) for f in files]
imgs = [plt.imread(path) for path in paths]

plt.imshow(imgs[3])
plt.imshow(imgs[2])

data = np.array(imgs)
data.shape

mean_img = np.mean(data, axis=0)
plt.imshow(mean_img.astype(np.uint8))

std_img = np.std(data, axis=0)
plt.imshow(std_img.astype(np.uint8))
plt.imshow(np.mean(std_img, axis=2).astype(np.uint8))

flattened = data.ravel()
flattened.shape
100 * 218 * 178 * 3

plt.close('all')
plt.hist(flattened.ravel(), 255)

bins = 20
fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True, sharex=True)

# Original Image Data
axs[0].hist((data[0]).ravel(), bins)
axs[0].set_title('img distribution')

# Mean Data
axs[1].hist(mean_img.ravel(), bins)
axs[1].set_title('mean distribution')

# Difference (Data[0] - Mean) distribution
axs[2].hist((data[0] - mean_img).ravel(), bins)
axs[2].set_title('(img - mean) distribution')

plt.show()

# Normalization
# Difference (Data[0] - Mean) distribution
axs[0].hist((data[0] - mean_img).ravel(), bins)
axs[0].set_title('(img - mean) distribution')

# stdev
axs[1].hist(std_img.ravel(), bins)
axs[1].set_title('std deviation distribution')

# normalized
axs[2].hist(((data[0] - mean_img) / std_img).ravel(), bins)
axs[2].set_title('((img - mean) / std_dev) distribution')

# A normalized distribution has what shape?
axs[2].set_xlim([-150, 150])
axs[2].set_xlim([-100, 100])
axs[2].set_xlim([-50, 50])
axs[2].set_xlim([-10, 10])
axs[2].set_xlim([-5, 5])

# What's the Normal Distribuion?

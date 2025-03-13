from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

# Load an example image
image = io.imread('./AI/Codes/Unsupervised_learning/Clustering/PNG_transparency_demonstration_1.png')
image = np.array(image, dtype=np.float64) / 255  # Normalize

# Reshape the image to be a long list of RGB values
w, h, d = image.shape
image_array = np.reshape(image, (w * h, d))

# Randomly sample a subset of the image
image_sample = shuffle(image_array, random_state=0)[:1000]

# Perform K-means clustering on the RGB values
kmeans = KMeans(n_clusters=16, random_state=0).fit(image_sample)
labels = kmeans.predict(image_array)

# Recreate the compressed image using the cluster centers (colors)
image_compressed = kmeans.cluster_centers_[labels].reshape(w, h, d)

# Display original and compressed images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title('Original Image')

ax[1].imshow(image_compressed)
ax[1].set_title('Compressed Image with K-means')
plt.show()

# Load data
import warnings
import cv2
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
import matplotlib.animation as animation
from PIL import Image  # Correct import for Pillow
from IPython.display import display

warnings.filterwarnings("ignore")

# Image Link (You can download and upload it): https://drive.google.com/file/d/16iMaYEGH-GgmqZfrCTw2vjYayllcgFcb/view?usp=sharing
# !gdown 16iMaYEGH-GgmqZfrCTw2vjYayllcgFcb

im = cv2.imread('./AI/Codes/Unsupervised_learning/Clustering/elephant.jpg') # Reads an image into BGR Format
im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
original_shape = im.shape
print(im.shape)

plt.imshow(im) # as RGB Format
plt.show()

all_pixels  = im.reshape((-1,3))

kmeans = KMeans(n_clusters=7)
kmeans.fit(all_pixels)

new_img = np.zeros((330*500,3),dtype='uint8')

colors = kmeans.cluster_centers_
lables = kmeans.labels_

# Iterate over the image
for ix in range(new_img.shape[0]):
    new_img[ix] = colors[lables[ix]]

new_img = new_img.reshape((original_shape))
plt.imshow(new_img)
plt.show()
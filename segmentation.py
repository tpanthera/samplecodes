import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('your_image.jpg')  # Replace 'your_image.jpg' with the actual image file

# Convert the image from BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array of pixels
pixels = image_rgb.reshape((-1, 3))

# Define the number of clusters (segments) you want
num_clusters = 3

# Apply k-means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(pixels)

# Get the labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Reshape the segmented image
segmented_image = centroids[labels].reshape(image_rgb.shape)

# Display the original and segmented images
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title('Segmented Image')

plt.show()

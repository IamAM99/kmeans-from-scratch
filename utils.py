"""Useful functions to use along with the k-means clustering algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans


def plot_error(X: np.ndarray, n_clusters: list):
    """Plot the clustering error per number of clusters. This is a method of finding
    the best number of clusters.

    Parameters
    ----------
    X : np.ndarray
        Input dataset.
    n_clusters : list
        A list of integers, for numbers of clusters to test.
    """
    # Calculate errors
    error = []
    for c in n_clusters:
        km = KMeans(c)
        _, _ = km.fit(X)
        error.append(km.error)

    # Plot the results
    plt.figure()
    plt.plot(n_clusters, error, "o-")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Error")
    plt.title("Error per Number of Clusters")
    plt.xticks(n_clusters)
    plt.grid(linestyle="--", linewidth=0.4)
    plt.show()


def compress_image(image: np.ndarray, n_colors: int) -> np.ndarray:
    """Compress an image by reducing the number of colors it contains using the
    k-means clustering algorithm.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    n_colors : int
        Number of colors that the output image should have.

    Returns
    -------
    np.ndarray
        A compressed image with reduced number of colors.
    """
    # make a clustering
    km = KMeans(n_colors)
    centers, labels = km.fit(image.reshape((-1, 3)).astype(np.float64))

    # convert values to int, to be compatible for images
    centers = centers.astype(np.uint8)

    # create the new image
    new_img = []
    for label in labels:
        new_img.append(centers[label])
    new_img = np.asarray(new_img).reshape(image.shape)

    return new_img

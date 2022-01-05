"""A class to implement k-means clustering algorithm
"""

import numpy as np
from typing import Tuple
from scipy.spatial.distance import cdist


class KMeans:
    """K-Means clustering class.
    """

    def __init__(self, k: int):
        """Initialize a KMeans object.

        Parameters
        ----------
        k : int
            Number of clusters.
        """
        self.k = k
        self.centers = None
        self.labels = None
        self.error = 0

    def fit(
        self, X: np.ndarray, max_iter: int = 300, tol: float = 0.0001
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit the dataset on the model.

        Parameters
        ----------
        X : np.ndarray
            The input dataset.
        max_iter : int, optional
            Maximum number of iterations, by default 300.
        tol : float, optional
            Maximum error tolerance, by default 0.0001.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (clustering centers, clustering labels)
        """
        # make sure that X is an array of valid shape
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(
                "Number of input dimentions should be exactly 2. The first dimention is the samples, and the second is the features."
            )

        # select random cluster points from the given dataset
        self.centers = X[np.random.choice(X.shape[0], self.k, replace=False), :]

        # calculate distances
        distances = cdist(X, self.centers, metric="euclidean")

        # determine labels
        self.labels = np.asarray([np.argmin(dist) for dist in distances])

        # updating process
        for _ in range(max_iter):
            new_centers = []
            for idx in range(self.k):
                temp_cent = X[idx == self.labels].mean(axis=0)
                new_centers.append(temp_cent)

            new_centers = np.vstack(new_centers)
            if np.linalg.norm(self.centers - new_centers, axis=1).max() < tol:
                break
            self.centers = np.copy(new_centers)

            distances = cdist(X, self.centers, metric="euclidean")
            self.labels = np.asarray([np.argmin(dist) for dist in distances])

        # calculate the final error
        self.error = np.sum(distances.min(axis=1)) / len(distances)

        return self.centers, self.labels

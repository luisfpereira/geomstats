"""K-medoids clustering.

Lead author: Hadi Zaatiti.
"""

import logging

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

import geomstats.backend as gs
from geomstats.learning._template import TransformerMixin


class RiemannianKMedoids(TransformerMixin, ClusterMixin, BaseEstimator):
    """Class for K-medoids clustering on manifolds.

    K-medoids algorithm using Riemannian manifolds.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    n_clusters : int
        Number of clusters (k value of k-medoids).
        Optional, default: 8.
    max_iter : int
        Maximum number of iterations.
        Optional, default: 100.
    init : str
        How to initialize cluster centers at the beginning of the algorithm. The
        choice 'random' will select training points as initial cluster centers
        uniformly at random.
        Optional, default: 'random'.
    n_jobs : int
        Number of jobs to run in parallel. `-1` means using all processors.
        Optional, default: 1.

    Notes
    -----
    * Required metric methods: `dist`, `dist_pairwise`.

    Example
    -------
    Available example on the Poincaré Ball and Hypersphere manifolds
    :mod:`examples.plot_kmedoids_manifolds`
    """

    # TODO: rename, this is not Riemannian
    # TODO: remove `dist_pairwise` from metric
    # TODO: implement example for Grassmanian

    def __init__(
        self,
        space,
        n_clusters=8,
        method="alternate",  # "pam" | "alternate"
        init="random",
        max_iter=100,
        n_jobs=1,
        store_dist_mat=True,
    ):
        self.space = space
        self.n_clusters = n_clusters
        self.method = method
        self.init = init
        self.store_dist_mat = store_dist_mat
        self.max_iter = max_iter
        self.n_jobs = n_jobs

        self.cluster_centers_ = None
        self.labels_ = None
        self.medoid_indices_ = None
        self.iter_ = None
        # TODO: add flag for storing this?
        if store_dist_mat:
            self.dist_mat_ = None

    def _initialize_medoids(self, dist_mat):
        """Select initial medoids when beginning clustering."""
        if self.init == "random":
            medoids = gs.random.choice(gs.arange(len(dist_mat)), self.n_clusters)
        else:
            logging.error("Unknown initialization method.")

        return medoids

    def _assign_cluster(self, dist_mat, medoid_indices):
        return gs.argmin(dist_mat[:, medoid_indices], axis=-1).T

    def _costs(self, dist_mat, labels, medoid_indices):
        indices = gs.arange(dist_mat.shape[0])

        if labels.ndim == 2:
            indices = indices[None, :]
            assigned_medoids = medoid_indices[
                gs.arange(medoid_indices.shape[0])[:, None], labels
            ]
        else:
            assigned_medoids = medoid_indices[labels]

        return dist_mat[indices, assigned_medoids]

    def _cost(self, dist_mat, labels, medoid_indices):
        return gs.sum(self._costs(dist_mat, labels, medoid_indices), axis=-1)

    def fit(self, X):
        """Provide cluster centers and data labels.

        Labels data by minimizing the distance between data points
        and cluster center chosen from the data points.
        Minimization is performed by swapping the cluster centers and data points.

        Parameters
        ----------
        X : array-like, shape=[n_samples, dim]
            Training data, where n_samples is the number of samples and
            dim is the number of dimensions.

        Returns
        -------
        self : object
            Returns self.
        """
        dist_mat = self.space.metric.dist_pairwise(X, n_jobs=self.n_jobs)
        if self.store_dist_mat:
            self.dist_mat_ = dist_mat

        # TODO: add callable
        medoid_indices = self._initialize_medoids(dist_mat)

        for iteration in range(self.max_iter):
            labels = self._assign_cluster(dist_mat, medoid_indices)

            if self.method == "alternate":
                new_medoid_indices = self._alternate_update(
                    dist_mat, labels, medoid_indices
                )
            else:  # TODO: do validation at init?
                # TODO: use labels?
                new_medoid_indices, _ = self._pam_update(
                    dist_mat,
                    labels,
                    medoid_indices,
                )

            if gs.all(new_medoid_indices == medoid_indices):
                break

            medoid_indices = new_medoid_indices

        else:
            medoid_indices = new_medoid_indices
            labels = self._assign_cluster(dist_mat, medoid_indices)
            logging.warning(
                "Maximum number of iteration reached before "
                "convergence. Consider increasing max_iter to "
                "improve the fit."
            )

        self.cluster_centers_ = X[medoid_indices]
        self.labels_ = labels
        self.medoid_indices_ = medoid_indices
        self.iter_ = iteration

        return self

    def _alternate_update(self, dist_mat, labels, medoid_indices):
        medoid_indices = gs.copy(medoid_indices)
        for cluster_label in range(self.n_clusters):
            cluster_points = gs.where(labels == cluster_label)[0]

            if len(cluster_points) == 0:
                logging.warning("One cluster is empty.")
                continue

            in_cluster_distances = dist_mat[cluster_points, cluster_points[..., None]]
            in_cluster_all_costs = gs.sum(in_cluster_distances, axis=1)

            min_cost_index = gs.argmin(in_cluster_all_costs)
            min_cost = in_cluster_all_costs[min_cost_index]

            current_medoid_index = gs.where(
                cluster_points == medoid_indices[cluster_label]
            )[0][0]
            current_cost = in_cluster_all_costs[current_medoid_index]

            if min_cost < current_cost:
                medoid_indices[cluster_label] = cluster_points[min_cost_index]

        return medoid_indices

    def _pam_update(self, dist_mat, labels, medoid_indices):
        current_cost = self._cost(dist_mat, labels, medoid_indices)

        n_points = dist_mat.shape[0]
        non_medoids = np.setdiff1d(gs.arange(n_points), medoid_indices)

        candidate_medoid_indices = []
        for medoid_pos, medoid in enumerate(medoid_indices):
            for candidate in non_medoids:
                new_medoid_indices = gs.copy(medoid_indices)
                new_medoid_indices[medoid_pos] = candidate
                candidate_medoid_indices.append(new_medoid_indices)

        candidate_medoid_indices = gs.array(candidate_medoid_indices)

        candidate_labels = self._assign_cluster(dist_mat, candidate_medoid_indices)
        candidate_costs = gs.sum(
            self._costs(dist_mat, candidate_labels, candidate_medoid_indices),
            axis=-1,
        )

        best_index = gs.argmin(candidate_costs)
        best_cost = candidate_costs[best_index]

        if best_cost < current_cost:
            return candidate_medoid_indices[best_index], candidate_labels[best_index]

        return medoid_indices, labels

    def predict(self, X):
        """Predict the closest cluster for each sample in X.

        Parameters
        ----------
        X : array-like, shape=[n_samples, dim]
            Training data, where n_samples is the number of samples and
            dim is the number of dimensions.

        Returns
        -------
        labels : array-like, shape=[n_samples,]
            Index of the cluster each sample belongs to.
        """
        labels = gs.zeros(len(X))

        # TODO: this can be done way better
        for point_index, point_value in enumerate(X):
            distances = gs.zeros(len(self.cluster_centers_))
            for cluster_index, cluster_value in enumerate(self.cluster_centers_):
                distances[cluster_index] = self.space.metric.dist(
                    point_value, cluster_value
                )

            labels[point_index] = gs.argmin(distances)

        return labels

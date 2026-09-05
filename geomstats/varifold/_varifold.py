"""Varifolds related machinery.

General framework is introduced in [KCC2017]_.
See [CCGGR2020]_ for details about kernels.
Implementation is based in pykeops (https://www.kernel-operations.io/keops/).
In particular, see
https://www.kernel-operations.io/keops/_auto_tutorials/surface_registration/plot_LDDMM_Surface.html#data-attachment-term # noqa
for implementation details.

References
----------
.. [KCC2017] Irene Kaltenmark, Benjamin Charlier, and Nicolas Charon.
    “A General Framework for Curve and Surface Comparison and Registration
    With Oriented Varifolds,” 3346–55, 2017.
    https://openaccess.thecvf.com/content_cvpr_2017/html/Kaltenmark_A_General_Framework_CVPR_2017_paper.html.
.. [CCGGR2020] Nicolas Charon, Benjamin Charlier, Joan Glaunès, Pietro Gori, and Pierre Roussillon.
    “Fidelity Metrics between Curves and Surfaces: Currents, Varifolds, and Normal
    Cycles.” In Riemannian Geometric Statistics in Medical Image Analysis,
    edited by Xavier Pennec, Stefan Sommer, and Tom Fletcher, 441–77.
    Academic Press, 2020. https://doi.org/10.1016/B978-0-12-814725-2.00021-2
"""

import abc

import geomstats.backend as gs
from geomstats._mesh import Surface

from ._device import resolve_device, to_cpu
from ._engine import resolve_engine
from .base import DiscreteMeasure
from .kernel import GaussianBinetPairing


class KernelInducedMetric(abc.ABC):
    """Metric induced by a kernel pairing.

    Parameters
    ----------
    pairing : Pairing
        Kernel pairing used to define the metric.
    """

    def __init__(self, pairing):
        self.pairing = pairing

    @abc.abstractmethod
    def transform(self, point):
        """Transform a point into the representation used by the pairing."""

    def scalar_product(self, point_a, point_b):
        """Compute the scalar product between two points.

        Parameters
        ----------
        point_a : object
            First point.
        point_b : object
            Second point.

        Returns
        -------
        scalar : float
            Scalar product between the points.
        """
        point_a = self.transform(point_a)
        point_b = self.transform(point_b)

        return to_cpu(self.pairing(point_a, point_b))

    def squared_dist(self, point_a, point_b):
        """Compute the squared distance between two points.

        Parameters
        ----------
        point_a : object
            First point.
        point_b : object
            Second point.

        Returns
        -------
        squared_dist : float
            Squared distance between the points.
        """
        point_a = self.transform(point_a)
        point_b = self.transform(point_b)

        sdist = (
            self.pairing(point_a, point_a)
            - 2 * self.pairing(point_a, point_b)
            + self.pairing(point_b, point_b)
        )
        return to_cpu(sdist)

    def dist(self, point_a, point_b):
        """Compute the distance between two points.

        Parameters
        ----------
        point_a : object
            First point.
        point_b : object
            Second point.

        Returns
        -------
        dist : float
            Distance between the points.
        """
        sq_dist = self.squared_dist(point_a, point_b)
        return gs.sqrt(sq_dist)

    def loss(self, target_point, target_faces=None):
        """Create a squared-distance loss to a target point.

        Parameters
        ----------
        target_point : Surface
            Target surface.
        target_faces : array-like, shape=[n_faces, 3], optional
            Combinatorial structure of the surface being optimized.
            If omitted, use ``target_point.faces``.

        Returns
        -------
        squared_dist : callable
            Function mapping surface vertices to their squared distance
            from ``target_point``.
        """
        if target_faces is None:
            target_faces = target_point.faces

        target_point = self.transform(target_point)
        kernel_target = self.pairing(target_point, target_point)

        def squared_dist(vertices):
            point = Surface(vertices, target_faces)
            point = self.transform(point)
            return (
                kernel_target
                - 2 * self.pairing(target_point, point)
                + self.pairing(point, point)
            )

        return squared_dist


class SurfaceMeasure(DiscreteMeasure):
    """Discrete measure representation of a surface.

    Parameters
    ----------
    centroids : array-like
        Face centroids.
    normals : array-like
        Face normals.
    areas : array-like
        Face areas.
    """

    def __init__(self, centroids, normals, areas):
        super().__init__(centroids, normals, areas)

    @property
    def centroids(self):
        """Face centroids."""
        return self.points

    @property
    def normals(self):
        """Face normals."""
        return self.features

    @property
    def areas(self):
        """Face areas."""
        return self.weights


class VarifoldMetric(KernelInducedMetric):
    """Varifold metric.

    Parameters
    ----------
    sigma : float
        Positive bandwidth parameter of the Gaussian kernel.
    engine : {"auto", "geomstats", "keops", "keops_genred", "keops_lazy"}
        Kernel computation engine. ``"auto"`` selects an engine automatically
        and ``"keops"`` is an alias for ``"keops_genred"``.
    device : {"auto", "cpu", "gpu"} or None
        Device for kernel computations. ``"auto"`` selects GPU when available
        and CPU otherwise. If ``None``, no device is selected.
    """

    def __init__(self, sigma, engine="auto", device="auto"):
        self.sigma = sigma

        self._engine = resolve_engine(engine)

        pairing = GaussianBinetPairing(sigma, engine=self._engine)
        super().__init__(pairing)

        self._device = resolve_device(device)

    def transform(self, point):
        """Convert a surface to its discrete measure representation.

        Parameters
        ----------
        point : Surface
            Surface-like object with attributes ``face_centroids``,
            ``face_normals``, and ``face_areas``.

        Returns
        -------
        measure : SurfaceMeasure
            Discrete surface measure on the configured device.
        """
        return SurfaceMeasure(
            point.face_centroids,
            point.face_normals,
            point.face_areas,
        ).to_device(self._device)

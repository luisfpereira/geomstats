"""Discrete measure representations and kernel pairings."""

import abc
import weakref

import geomstats.backend as gs

from ._device import get_device, to_device


class DiscreteMeasure:
    """Discrete weighted measure.

    Parameters
    ----------
    points : array-like, shape=[n_samples, point_dim]
        Support points of the measure.
    features : array-like, shape=[n_samples, feature_dim]
        Features associated with the support points.
    weights : array-like, shape=[n_samples]
        Weights of the support points.
    """

    def __init__(self, points, features, weights):
        self.points = points
        self.features = features
        self.weights = weights

    def _new(self, points, features, weights):
        return self.__class__(points, features, weights)

    @property
    def device(self):
        """Device on which the measure is stored."""
        return get_device(self.points)

    def to_device(self, device):
        """Move the measure to a device.

        Parameters
        ----------
        device : {"cpu", "gpu"} or None
            Target device. If ``None``, return the measure unchanged.

        Returns
        -------
        measure : DiscreteMeasure
            Measure on the requested device.
        """
        if device is None or device == self.device:
            return self

        return self._new(
            points=to_device(self.points, device),
            features=to_device(self.features, device),
            weights=to_device(self.weights, device),
        )


class Pairing(abc.ABC):
    """Kernel pairing between discrete measures.

    Parameters
    ----------
    cache : bool
        Whether to cache self-pairings.
    """

    def __init__(self, cache=True):
        self._cache = cache
        self._self_pair_cache = weakref.WeakKeyDictionary()

    def reset_cache(self):
        """Clear cached self-pairings."""
        self._self_pair_cache.clear()

    def self_pair(self, point):
        """Evaluate the self-pairing of a point.

        Parameters
        ----------
        point : DiscreteMeasure
            Point to pair with itself.

        Returns
        -------
        scalar : float
            Self-pairing of the point.
        """
        if not self._cache:
            return self(point, point)

        try:
            return self._self_pair_cache[point]
        except KeyError:
            value = self(point, point)
            self._self_pair_cache[point] = value
            return value

    def __call__(self, point_a, point_b):
        """Evaluate the pairing.

        Parameters
        ----------
        point_a : DiscreteMeasure
            First measure.
        point_b : DiscreteMeasure
            Second measure.

        Returns
        -------
        scalar : float
            Pairing between the measures.
        """
        return gs.sum(self.kernel_prod(point_a, point_b) * point_a.weights)

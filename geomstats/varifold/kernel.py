"""Kernel pairings."""

import geomstats.backend as gs

from .base import Pairing

if gs.__name__.endswith("pytorch"):
    import torch

    _compile = torch.compile

else:

    def _compile(fn):
        return fn


def GaussianBinetPairing(sigma, engine, cache=True):
    r"""Instantiate a Gaussian-Binet kernel pairing.

    The kernel is defined by

    .. math::

        K(x, y, u, v)
        = \exp(-\|x - y\|^2 / \sigma^2) \langle u, v \rangle^2.

    Parameters
    ----------
    sigma : float
        Positive bandwidth parameter of the Gaussian kernel.
    engine : {"geomstats", "keops_genred", "keops_lazy"}
        Kernel computation engine.

        - ``"geomstats"``: Dense implementation using the current backend.
        - ``"keops_genred"``: KeOps implementation using Genred reductions.
        - ``"keops_lazy"``: KeOps implementation using LazyTensor reductions.
    cache : bool
        Whether to cache self-pairings.

    Returns
    -------
    Pairing
        Gaussian-Binet kernel pairing.

    Notes
    -----
    The dense ``"geomstats"`` implementation materializes pairwise matrices
    and is memory-bound for large inputs. KeOps implementations avoid
    materializing the full kernel matrix and are better suited to large-scale
    computations.
    """
    if engine == "geomstats":
        return _GaussianBinetPairing(sigma=sigma, cache=cache)

    if engine == "keops_genred":
        import geomstats.varifold.keops.genred as gkeops

        return gkeops.GaussianBinetPairing(sigma, cache=cache)

    if engine == "keops_lazy":
        import geomstats.varifold.keops.lazy as lkeops

        return lkeops.SurfaceKernelPairing(
            lkeops.GaussianBinetKernel(sigma=sigma), cache=cache
        )

    raise ValueError(f"Unknown engine: {engine}")


class _GaussianBinetPairing(Pairing):
    r"""Instantiate a Gaussian-Binet kernel pairing.

    This pairing is defined by

    .. math::

        K(x, y, u, v) = exp(-||x - y||^2 / sigma^2) <u, v>^2

    Parameters
    ----------
    sigma : float
        Positive bandwidth parameter of the Gaussian kernel.
    cache : bool
        Whether to cache self-pairings.

    Notes
    -----
    It materializes pairwise matrices and is memory-bound for large inputs.
    """

    def __init__(self, sigma, cache=True):
        super().__init__(cache=cache)

        def _kernel(x, y, u, v):
            x_norm2 = gs.sum(x**2, axis=1)[:, None]
            y_norm2 = gs.sum(y**2, axis=1)[None, :]

            dist2 = x_norm2 + y_norm2 - 2 * (x @ y.T)
            K_xy = gs.exp(-dist2 / sigma**2)

            uv = u @ v.T

            return K_xy * uv**2

        self._kernel = _compile(_kernel)

    def kernel_prod(self, point_a, point_b):
        """Apply the kernel operator to the second measure's weights.

        Parameters
        ----------
        point_a : DiscreteMeasure
            First measure.
        point_b : DiscreteMeasure
            Second measure.

        Returns
        -------
        kernel_prod : array-like
            Kernel reduction against ``point_b.weights``.
        """
        return (
            self._kernel(
                point_a.points,
                point_b.points,
                point_a.features,
                point_b.features,
            )
            @ point_b.weights
        )

"""Kernels and kernel pairings using KeOps genred."""

import geomstats.backend as gs
from geomstats.varifold.base import Pairing

if gs.__name__.endswith("pytorch"):
    from pykeops.torch import Genred
else:
    from pykeops.numpy import Genred

from .._device import to_device
from ._device import _keops_backend


def GaussianKernel(sigma):
    r"""Gaussian kernel.

    .. math::

        K(x, y)=e^{-\|x-y\|^2 / \sigma^2}

    Parameters
    ----------
    sigma : float
        Kernel parameter.
    """
    expr = Genred(
        "Exp(-SqDist(x,y)*a)",
        [
            "a=Pm(1)",
            "x=Vi(3)",
            "y=Vj(3)",
        ],
        reduction_op="Sum",
        axis=1,
    )
    a_param = 1 / gs.array([sigma]) ** 2

    def kernel_eval(point_a, point_b):
        return expr(a_param, point_a, point_b)

    return kernel_eval


def CauchyKernel(sigma):
    r"""Cauchy kernel.

    .. math::

        K(x, y)=\frac{1}{1+\|x-y\|^2 / \sigma^2}

    Parameters
    ----------
    sigma : float
        Kernel parameter.
    """
    expr = Genred(
        "IntCst(1)/(IntCst(1)+SqDist(x,y)*a)",
        [
            "a=Pm(1)",
            "x=Vi(3)",
            "y=Vj(3)",
        ],
        reduction_op="Sum",
        axis=1,
    )
    a_param = 1 / gs.array([sigma]) ** 2

    def kernel_eval(point_a, point_b):
        return expr(a_param, point_a, point_b)

    return kernel_eval


def LinearKernel():
    r"""Linear kernel.

    .. math::

        K(u, v) = \langle u, v \rangle
    """
    expr = Genred(
        "(u|v)",
        [
            "u=Vi(3)",
            "v=Vj(3)",
        ],
        reduction_op="Sum",
        axis=1,
    )

    def kernel_eval(point_a, point_b):
        return expr(point_a, point_b)

    return kernel_eval


def BinetKernel():
    r"""Binet kernel.

    .. math::

        K(u, v) = \langle u, v \rangle^2
    """
    expr = Genred(
        "Square((u|v))",
        [
            "u=Vi(3)",
            "v=Vj(3)",
        ],
        reduction_op="Sum",
        axis=1,
    )

    def kernel_eval(point_a, point_b):
        return expr(point_a, point_b)

    return kernel_eval


def OrientedGaussianKernel(sigma=1.0):
    r"""Gaussian kernel restricted to the hypersphere.

    .. math::

        K(u, v)=e^{2 (\langle u, v \rangle / - 1) / \sigma^2}

    Parameters
    ----------
    sigma : float
        Kernel parameter.
    """
    expr = Genred(
        "Exp(IntCst(2)*b*((u|v)-IntCst(1)))",
        [
            "b=Pm(1)",
            "u=Vi(3)",
            "v=Vj(3)",
        ],
        reduction_op="Sum",
        axis=1,
    )

    a_param = 1 / gs.array([sigma]) ** 2

    def kernel_eval(point_a, point_b):
        return expr(a_param, point_a, point_b)

    return kernel_eval


def UnorientedGaussianKernel(sigma=1.0):
    r"""Gaussian kernel restricted to the hypersphere.

    .. math::

        K(u, v)=e^{2 (\langle u, v \rangle ^2 - 1) / \sigma^2 }


    Parameters
    ----------
    sigma : float
        Kernel parameter.
    """
    expr = Genred(
        "Exp(IntCst(2)*b*(Square((u|v))-IntCst(1)))",
        [
            "b=Pm(1)",
            "u=Vi(3)",
            "v=Vj(3)",
        ],
        reduction_op="Sum",
        axis=1,
    )

    a_param = 1 / gs.array([sigma]) ** 2

    def kernel_eval(point_a, point_b):
        return expr(a_param, point_a, point_b)

    return kernel_eval


class GaussianBinetPairing(Pairing):
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
    """

    def __init__(self, sigma, cache=True):
        super().__init__(cache=cache)
        self._expr = Genred(
            "Exp(-SqDist(x,y)*a)*Square((u|v))*b",
            [
                "a=Pm(1)",
                "x=Vi(3)",
                "y=Vj(3)",
                "u=Vi(3)",
                "v=Vj(3)",
                "b=Vj(1)",
            ],
            reduction_op="Sum",
            axis=1,
        )
        self._a_param = 1 / gs.array([sigma]) ** 2

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
        device = point_a.device
        a_param = to_device(self._a_param, device)

        return self._expr(
            a_param,
            point_a.points,
            point_b.points,
            point_a.features,
            point_b.features,
            point_b.weights,
            backend=_keops_backend(device),
        )

import abc

import geomstats.backend as gs
from geomstats.geometry.connection import Connection
from geomstats.geometry.riemannian_metric import RiemannianMetric


class PotentialFunction(abc.ABC):
    # TODO: pay attention to create_graph

    @classmethod
    def from_func(cls, func):
        # TODO: this interface may be weird when inheriting
        class _PotentialFunctionFromFunction(cls):
            def __call__(self, point):
                return func(point)

        return _PotentialFunctionFromFunction()

    @abc.abstractmethod
    def __call__(self, point):
        pass

    def grad(self, point):
        # TODO: need to get create_graph
        _, grad = gs.autodiff.value_and_grad(self.__call__)(point)
        return grad

    def hessian(self, point):
        return gs.autodiff.hessian(self.__call__, create_graph=True)(point)

    def third_derivative(self, point):
        return gs.autodiff.jacobian(self.hessian)(point)


class Divergence(abc.ABC):
    @classmethod
    def from_func(cls, func):
        # TODO: this interface may be weird when inheriting
        class _StatisticalDivergenceFromFunction(cls):
            def __call__(self, point_a, point_b):
                return func(point_a, point_b)

        return _StatisticalDivergenceFromFunction()

    @abc.abstractmethod
    def __call__(self, point_a, point_b):
        pass

    def _hessian(self, point_a, point_b):
        # TODO: issue with Euclidean... symmetry?

        from torch.autograd.functional import hessian as _torch_hessian

        return _torch_hessian(
            self.__call__,
            inputs=(point_a, point_b),
            create_graph=True,
        )

    def first_first_derivative(self, point_a, point_b):
        return self._hessian(point_a, point_b)[0][1]

    def first_first_derivative_at_point(self, point):
        return self.first_first_derivative(point, point)

    def second_zeroth_derivative(self, point_a, point_b):
        return self._hessian(point_a, point_b)[0][0]

    def zeroth_second_derivative(self, point_a, point_b):
        return self._hessian(point_a, point_b)[1][1]

    def second_first_derivative(self, point_a, point_b):
        from torch.autograd.functional import jacobian as _torch_jacobian

        return _torch_jacobian(
            self.second_zeroth_derivative,
            inputs=(point_a, point_b),
        )[1]

    def second_first_derivative_at_point(self, point):
        return self.second_first_derivative(point, point)

    def first_second_derivative(self, point_a, point_b):
        from torch.autograd.functional import jacobian as _torch_jacobian

        return _torch_jacobian(
            self.zeroth_second_derivative,
            inputs=(point_a, point_b),
        )[0]

    def first_second_derivative_at_point(self, point):
        """First-second derivative.

        Equivalent to take second-first derivative of dual
        divergence.
        """
        return self.first_second_derivative(point, point)


class DualDivergence(Divergence):
    def __new__(cls, divergence, **kwargs):
        if (
            isinstance(divergence, BregmanDivergence)
            and cls is not DualBregmanDivergence
        ):
            return DualBregmanDivergence(divergence, **kwargs)

        return super().__new__(cls)

    def __init__(self, divergence, **kwargs):
        super().__init__(**kwargs)
        self.divergence = divergence

    def __call__(self, point_a, point_b):
        return self.divergence(point_b, point_a)


class BregmanDivergence(Divergence):
    # TODO: bring third derivative in. how?

    def __init__(self, potential_function, **kwargs):
        super().__init__(**kwargs)
        self.potential_function = potential_function

    def __call__(self, point_a, point_b):
        f_a = self.potential_function(point_a)
        f_b = self.potential_function(point_b)

        grad_at_b = self.potential_function.grad(point_b)
        diff = point_a - point_b

        return f_a - f_b - gs.dot(grad_at_b, diff)

    def first_first_derivative_at_point(self, point):
        return -self.potential_function.hessian(point)


class DualBregmanDivergence(DualDivergence, BregmanDivergence):
    def __init__(self, divergence):
        super().__init__(
            divergence=divergence, potential_function=divergence.potential_function
        )
        self.divergence = divergence


class DivergenceInducedRiemannianMetric(RiemannianMetric):
    def __init__(self, space, divergence):
        super().__init__(space)
        self.divergence = divergence

    def metric_matrix(self, base_point):
        """Metric matrix at the tangent space at a base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        return -self.divergence.first_first_derivative_at_point(base_point)


class UnnamedConnection(Connection):
    # TODO: find a better name

    def christoffels(self, base_point):
        r"""Compute the (second kind) Christoffel symbols of the divergence connection.

        Compute the (second kind) Christoffel symbols of the divergence connection
        :math:`\nabla^D` at the tangent space of the base point.

        .. math::
            \Gamma^D_{ij}^k = g^{kl} \Gamma^D_{ijl}

        where :math:`g^{kl}` is the cometric matrix of the divergence induced metric.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        matrix : array-like, shape=[..., dim, dim, dim]
            Second kind Christoffel symbols of the divergence connection.
        """
        # TODO: review docstrings
        # TODO: need to figure out where to put this; also used in CubicTensorInducedConnection
        cometric_matrix = self._space.metric.cometric_matrix(base_point)
        first_kind_christoffels = self.first_kind_christoffels(base_point)
        second_kind_christoffels = gs.einsum(
            "...lk, ...ijl -> ...kij", cometric_matrix, first_kind_christoffels
        )
        return second_kind_christoffels


class DivergenceInducedConnection(UnnamedConnection):
    def __init__(self, space, divergence):
        # NB: assumes space has a metric
        super().__init__(space)
        self.divergence = divergence

    def first_kind_christoffels(self, base_point):
        r"""Compute the first kind Christoffel symbols of the divergence connection.

        Compute the first kind Christoffel symbols of the divergence connection
        :math:`\nabla^D` at the tangent space of the base point.

        .. math::
            \Gamma^D_{i j k} =
            -1 \cdot \frac{\partial^2}{\partial x^i \partial x^j}
                \frac{\partial}{\partial y^k} D(x, y) \bigg|_{x=y}

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        matrix : array-like, shape=[..., dim, dim, dim]
            First kind Christoffel symbols of the divergence connection.
        """
        # TODO: review docstrings
        return -self.divergence.second_first_derivative_at_point(base_point)


class CubicTensorInducedConnection(UnnamedConnection):
    # TODO: make a variant from connection-pair for speed

    def __init__(self, space, cubic_tensor, param=1.0):
        # NB: assumes space has a metric
        super().__init__(space)
        self.cubic_tensor = cubic_tensor
        self.param = param

    def first_kind_christoffels(self, base_point):
        r"""Compute the first kind Christoffel symbols of the divergence connection.

        Compute the first kind Christoffel symbols of the divergence connection
        :math:`\nabla^D` at the tangent space of the base point.

        .. math::
            \Gamma^D_{i j k} =
            -1 \cdot \frac{\partial^2}{\partial x^i \partial x^j}
                \frac{\partial}{\partial y^k} D(x, y) \bigg|_{x=y}

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        matrix : array-like, shape=[..., dim, dim, dim]
            First kind Christoffel symbols of the divergence connection.
        """
        # TODO: review docstrings
        # TODO: decide convention on param sign
        return self._space.metric.first_kind_christoffels(
            base_point
        ) - 0.5 * self.param * self.cubic_tensor(base_point)


class ConnectionPair:
    # can induce a cubic tensor
    # can be induced by a cubic tensor

    # TODO: may need to split class

    # TODO: call it conjugate connection pair instead?

    def __init__(self, primal_connection, dual_connection, cubic_tensor=None):
        # TODO: will need to improve API

        self.primal = primal_connection
        self.dual = dual_connection

        # NB: all dualistic structures have an associated cubic tensor
        # (that they induce or from which they're induced)
        if cubic_tensor is None:
            cubic_tensor = DualisticStructureInducedCubicTensor(self)
        self.cubic_tensor = cubic_tensor

    @classmethod
    def from_divergence(cls, space, divergence, dual_divergence=None):
        primal_connection = DivergenceInducedConnection(space, divergence)

        if dual_divergence is None:
            dual_divergence = DualDivergence(divergence)
        dual_connection = DivergenceInducedConnection(space, dual_divergence)

        return cls(primal_connection, dual_connection)

    @classmethod
    def from_cubic_tensor(cls, space, cubic_tensor):
        primal_connection = CubicTensorInducedConnection(space, cubic_tensor, param=1.0)
        dual_connection = CubicTensorInducedConnection(space, cubic_tensor, param=-1.0)

        return cls(primal_connection, dual_connection, cubic_tensor=cubic_tensor)


class CubicTensor(abc.ABC):
    # TODO: get access to space?
    @abc.abstractmethod
    def __call__(self, base_point):
        """Totally symmetric cubic tensor.

        .. math::

            T = \nabla^* - \nabla
        """


class DualisticStructureInducedCubicTensor(CubicTensor):
    def __init__(self, connection_pair):
        # TODO: add param
        self.connection_pair = connection_pair

    def __call__(self, base_point):
        """Totally symmetric cubic tensor.

        aka skewness tensor of a dualistic structure.

        aka Amari-Chentsov tensor?

        .. math::

            T = \nabla^* - \nabla
        """
        # TODO: need to check sign
        return self.connection_pair.primal.first_kind_christoffels(
            base_point
        ) - self.connection_pair.dual.first_kind_christoffels(base_point)


class DualisticStructureInducingCubicTensor(CubicTensor, abc.ABC):
    pass


class PotentialFunctionInducedCubicTensor(DualisticStructureInducingCubicTensor):
    def __init__(self, potential_function):
        super().__init__
        self.potential_function = potential_function

    def __call__(self, base_point):
        return self.potential_function.third_derivative(base_point)

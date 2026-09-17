import geomstats.backend as gs
from geomstats.vectorization import get_batch_shape

from .manifold import ManifoldTestCase
from .mixins import ProjectionTestCaseMixins


class LevelSetTestCase(ProjectionTestCaseMixins, ManifoldTestCase):
    # TODO: need to develop ``intrinsic_after_extrinsic``
    # and ``extrinsic_after_intrinsic``
    # TODO: need to develop `intrinsic_after_extrinsic` and `extrinsic_after_intrinsic`
    # TODO: class to handle `extrinsinc-intrinsic` mixins?

    def test_submersion(self, point, expected, atol=gs.atol):
        submersed_point = self.space.submersion(point)
        self.assertAllClose(submersed_point, expected, atol=atol)

    def test_submersion_is_zero(self, point, submersion_shape, atol=gs.atol):
        # TODO: keep?
        batch_shape = get_batch_shape(self.space.point_ndim, point)
        expected = gs.zeros(batch_shape + submersion_shape)

        self.test_submersion(point, expected, atol)

    def test_tangent_submersion(self, vector, point, expected, atol=gs.atol):
        submersed_vector = self.space.tangent_submersion(vector, point)
        self.assertAllClose(submersed_vector, expected, atol=atol)

    def test_tangent_submersion_is_zero(
        self, tangent_vector, point, tangent_submersion_shape, atol
    ):
        # TODO: keep?
        batch_shape = get_batch_shape(self.space.point_ndim, tangent_vector, point)
        expected = gs.zeros(batch_shape + tangent_submersion_shape)

        self.test_tangent_submersion(tangent_vector, point, expected, atol)

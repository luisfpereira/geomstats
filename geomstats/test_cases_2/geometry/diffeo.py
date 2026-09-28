import pytest

import geomstats.backend as gs
from geomstats.test.test_case import TestCase
from geomstats.vectorization import get_batch_shape


class DiffeoTestCase(TestCase):
    def test_diffeomorphism(self, base_point, expected, atol=gs.atol):
        res = self.diffeo(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_diffeomorphism_belongs(self, base_point, atol=gs.atol):
        image_point = self.diffeo(base_point)

        belongs = self.image_space.belongs(image_point, atol=atol)

        expected_shape = get_batch_shape(
            self.domain_space.point_ndim,
            base_point,
        )
        expected = gs.ones(expected_shape, dtype=bool)

        self.assertAllEqual(belongs, expected)

    def test_inverse(self, image_point, expected, atol=gs.atol):
        res = self.diffeo.inverse(image_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_inverse_belongs(self, image_point, atol=gs.atol):
        point = self.diffeo.inverse(image_point)

        belongs = self.domain_space.belongs(point, atol=atol)

        expected_shape = get_batch_shape(
            self.image_space.point_ndim,
            image_point,
        )
        expected = gs.ones(expected_shape, dtype=bool)

        self.assertAllEqual(belongs, expected)

    def test_inverse_after_diffeomorphism(self, base_point, atol=gs.atol):
        image_point = self.diffeo(base_point)
        base_point_ = self.diffeo.inverse(image_point)

        self.assertAllClose(base_point_, base_point, atol=atol)

    def test_diffeomorphism_after_inverse(self, image_point, atol=gs.atol):
        point = self.diffeo.inverse(image_point)
        image_point_ = self.diffeo(point)

        self.assertAllClose(image_point_, image_point, atol=atol)

    def test_tangent(
        self, tangent_vec, expected, atol=gs.atol, base_point=None, image_point=None
    ):
        res = self.diffeo.tangent(
            tangent_vec, base_point=base_point, image_point=image_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_tangent_is_tangent(self, tangent_vec, base_point, atol=gs.atol):
        image_tangent_vec = self.diffeo.tangent(tangent_vec, base_point)
        image_point = self.diffeo(base_point)

        is_tangent = self.image_space.is_tangent(
            image_tangent_vec, image_point, atol=atol
        )

        expected_shape = get_batch_shape(
            self.domain_space.point_ndim,
            tangent_vec,
            base_point,
        )
        expected = gs.ones(expected_shape, dtype=bool)

        self.assertAllEqual(is_tangent, expected)

    @pytest.mark.api
    def test_tangent_with_image_point(self, tangent_vec, base_point, atol=gs.atol):
        image_point = self.diffeo(base_point)

        image_tangent_vec = self.diffeo.tangent(tangent_vec, base_point)
        image_tangent_vec_ = self.diffeo.tangent(tangent_vec, image_point=image_point)

        self.assertAllClose(image_tangent_vec, image_tangent_vec_, atol=atol)

    def test_inverse_tangent(
        self,
        image_tangent_vec,
        expected,
        atol=gs.atol,
        image_point=None,
        base_point=None,
    ):
        res = self.diffeo.inverse_tangent(
            image_tangent_vec,
            image_point=image_point,
            base_point=base_point,
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_inverse_tangent_is_tangent(
        self, image_tangent_vec, image_point, atol=gs.atol
    ):
        tangent_vec = self.diffeo.inverse_tangent(image_tangent_vec, image_point)
        base_point = self.diffeo.inverse(image_point)

        is_tangent = self.domain_space.is_tangent(tangent_vec, base_point, atol=atol)

        expected_shape = get_batch_shape(
            self.image_space.point_ndim,
            image_tangent_vec,
            image_point,
        )
        expected = gs.ones(expected_shape, dtype=bool)

        self.assertAllEqual(is_tangent, expected)

    @pytest.mark.api
    def test_inverse_tangent_with_base_point(
        self, image_tangent_vec, image_point, atol=gs.atol
    ):
        base_point = self.diffeo.inverse(image_point)

        tangent_vec = self.diffeo.inverse_tangent(image_tangent_vec, image_point)
        tangent_vec_ = self.diffeo.inverse_tangent(
            image_tangent_vec, base_point=base_point
        )
        self.assertAllClose(tangent_vec, tangent_vec_, atol=atol)

    def test_inverse_tangent_after_tangent(self, tangent_vec, base_point, atol=gs.atol):
        image_tangent_vec = self.diffeo.tangent(tangent_vec, base_point)
        image_point = self.diffeo(base_point)

        tangent_vec_ = self.diffeo.inverse_tangent(image_tangent_vec, image_point)
        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    def test_tangent_after_inverse_tangent(
        self, image_tangent_vec, image_point, atol=gs.atol
    ):
        tangent_vec = self.diffeo.inverse_tangent(image_tangent_vec, image_point)
        base_point = self.diffeo.inverse(image_point)

        image_tangent_vec_ = self.diffeo.tangent(tangent_vec, base_point)
        self.assertAllClose(image_tangent_vec_, image_tangent_vec, atol=atol)

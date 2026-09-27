import operator

from polpo.testing.data import FiberBundleCaseData, LazyValue

from geomstats.vectorization import repeat_point


class FiberBundleTestData(FiberBundleCaseData):
    def riemannian_submersion_belongs_to_base_test_data(self):
        return self.generate_random_data(
            arg_names="point",
        )

    def lift_belongs_to_total_space_test_data(self):
        return self.generate_random_data(
            arg_names="point",
            space="base",
        )

    def riemannian_submersion_after_lift_test_data(self):
        return self.generate_random_data(
            arg_names="point",
            space="base",
        )

    def tangent_riemannian_submersion_is_tangent_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def log_after_align_is_horizontal_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "point"))

    def horizontal_projection_is_horizontal_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def vertical_projection_is_vertical_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def tangent_riemannian_submersion_after_vertical_projection_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def horizontal_lift_is_horizontal_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec"),
            space="base",
        )

    def tangent_riemannian_submersion_after_horizontal_lift_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec"),
            space="base",
        )


class FiberBundleVecTestData(FiberBundleCaseData):
    def riemannian_submersion_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("point"),
        )

    def lift_vec_test_data(self):
        return self.generate_vectorization_base_space_data(
            arg_names=("point"),
            space="base",
        )

    def tangent_riemannian_submersion_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def align_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "point"),
        )

    def horizontal_projection_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def vertical_projection_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def is_horizontal_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def is_vertical_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec"),
        )

    def horizontal_lift_vec_test_data(self):
        return self.generate_vectorization_base_space_data(
            arg_names=("base_point", "tangent_vec"),
            space="base",
        )

    def integrability_tensor_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b"),
        )

    def integrability_tensor_derivative_vec_test_data(self):
        datum = self._generate_lifted_random_datum(
            point_name="base_point",
            horizontal_names=("horizontal_vec_x", "horizontal_vec_y"),
            tangent_names=("nabla_x_y", "tangent_vec_e", "nabla_x_e"),
            n_points=1,
        )

        expected = LazyValue(
            lambda **kwargs: self.total_space.fiber_bundle.integrability_tensor_derivative(
                **kwargs
            ),
            **datum,
        )

        expected_nabla_x_a_y_e = LazyValue(
            operator.itemgetter(0),
            expected,
        )
        expected_a_y_e = LazyValue(
            operator.itemgetter(1),
            expected,
        )

        n_reps = 2

        vectorized_datum = {
            name: LazyValue(
                repeat_point,
                value,
                n_reps=n_reps,
                expand=True,
            )
            for name, value in datum.items()
        }

        vectorized_datum.update(
            expected_nabla_x_a_y_e=LazyValue(
                repeat_point,
                expected_nabla_x_a_y_e,
                n_reps=n_reps,
            ),
            expected_a_y_e=LazyValue(
                repeat_point,
                expected_a_y_e,
                n_reps=n_reps,
            ),
        )

        return [vectorized_datum]

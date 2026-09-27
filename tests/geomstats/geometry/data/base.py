from .manifold import ManifoldTestData
from .mixins import ProjectionMixinsTestData


class LevelSetTestData(ProjectionMixinsTestData, ManifoldTestData):
    def submersion_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="point",
        )

    def tangent_submersion_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("point", "vector"),
        )

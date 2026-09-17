from .manifold import ManifoldTestData, ManifoldVecTestData
from .mixins import ProjectionMixinsTestData, ProjectionMixinsVecTestData


class LevelSetTestData(ProjectionMixinsTestData, ManifoldTestData):
    pass


class LevelSetVecTestData(ProjectionMixinsVecTestData, ManifoldVecTestData):
    def submersion_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="point",
        )

    def tangent_submersion_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("point", "vector"),
        )

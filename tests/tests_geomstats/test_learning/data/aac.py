from geomstats.test.test_case import np_backend

from ._base import BaseEstimatorTestData, MeanEstimatorMixinsTestData

IS_NOT_NP = not np_backend()


class AACFrechetMeanTestData(MeanEstimatorMixinsTestData, BaseEstimatorTestData):
    pass


class AACGGPCATestData(BaseEstimatorTestData):
    trials = 5

    tolerances = {
        "fit_geodesic_points": {"atol": 1e-4},
    }

    def fit_geodesic_points_test_data(self):
        return self.generate_random_data()


class AACRegressionTestData(BaseEstimatorTestData):
    def fit_and_predict_constant_test_data(self):
        return self.generate_random_data()

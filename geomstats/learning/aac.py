"""Align All and Compute for Graph Space.

Lead author: Anna Calissano.
"""

from geomstats.errors import check_parameter_accepted_values
from geomstats.learning.frechet_mean import AACFrechetMean
from geomstats.learning.pca import AACGGPCA
from geomstats.learning.regression import AACRegression


class AAC:
    r"""Class for Align all and Compute algorithm on Graph Space.

    The Align All and Compute (AAC) algorithm is introduced in [CFV2020]_ and it
    allows to compute different statistical estimators: the Frechet Mean, the
    Generalized Geodesic Principal components and the Regression for a set of labeled or
    unlabeled graphs.
    The idea is to optimally aligned the graphs to the current
    estimator using the correct alignment technique and compute the current estimation
    using the geometrical property of the total space, i.e., the Euclidean space of
    adjacency matrices.

    Parameters
    ----------
    space : GraphSpace
        Graph space total space with a quotient structure.
    estimate : str
        Desired estimator. One of the following:

        - "frechet_mean": Frechet Mean estimation [CFV2020]_
        - "ggpca": Generalized Geodesic Principal Components [CFV2020]_
        - "regression": Graph-on-vector regression model [CFV2022]_

    Examples
    --------
    Available example on Graph Space:
    :mod:`notebooks.19_practical_methods__aac`

    Available example on Graph Space with real world data:
    :mod:`notebooks.20_real_world_application__graph_space`

    References
    ----------
    .. [CFV2020]  Calissano, A., Feragen, A., Vantini, S.
        “Graph Space: Geodesic Principal Components for a Population of
        Network-valued Data.” Mox report 14, 2020.
        https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
    .. [CFV2022]  Calissano, A., Feragen, A., Vantini, S.
        “Graph-valued regression: prediction of unlabelled networks in a non-Euclidean
        Graph Space.”Journal of Multivariate Analysis 190 - 104950, (2022).
        https://doi.org/10.1016/j.jmva.2022.104950.
    """

    MAP_ESTIMATE = {
        "frechet_mean": AACFrechetMean,
        "ggpca": AACGGPCA,
        "regression": AACRegression,
    }

    def __new__(cls, space, *, estimate="frechet_mean", **kwargs):
        """Class for Align all and Compute algorithm on Graph Space."""
        check_parameter_accepted_values(
            estimate, "estimate", list(cls.MAP_ESTIMATE.keys())
        )

        return cls.MAP_ESTIMATE[estimate](space, **kwargs)

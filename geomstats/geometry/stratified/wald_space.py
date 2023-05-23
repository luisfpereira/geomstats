r"""Classes for the Wald Space and elements therein of class Wald and helper classes.

Class ``Topology``.
A structure is a partition into non-empty sets of the set :math:`\{0,\dots,n-1\}`,
together with a set of splits for each element of the partition, where every split is a
two-set partition of the respective element.
A structure basically describes a phylogenetic forest, where each set of splits gives
the structure of the tree with the labels of the corresponding element of the partition.

Class ``Wald``.
A wald is essentially a phylogenetic forest with weights between zero and one on the
edges. The forest structure is stored as a ``Topology`` and the edge weights are an
array of length that is equal to the total number of splits in the structure. These
elements are the points in Wald space and other phylogenetic forest spaces, like BHV
space, although the partition is just the whole set of labels in this case.

Class ``WaldSpace``.
A topological space. Points in Wald space are instances of the class :class:`Wald`:
phylogenetic forests with edge weights between 0 and 1.
In particular, Wald space is a stratified space, each stratum is called grove.
The highest dimensional groves correspond to fully resolved or binary trees.
The topology is obtained from embedding wälder into the ambient space of strictly
positive :math:`n\times n` symmetric matrices, implemented in the
class :class:`spd.SPDMatrices`.


Lead author: Jonas Lueg


References
----------
[Garba21]_  Garba, M. K., T. M. W. Nye, J. Lueg and S. F. Huckemann.
            "Information geometry for phylogenetic trees"
            Journal of Mathematical Biology, 82(3):19, February 2021a.
            https://doi.org/10.1007/s00285-021-01553-x.
[Lueg21]_   Lueg, J., M. K. Garba, T. M. W. Nye, S. F. Huckemann.
            "Wald Space for Phylogenetic Trees."
            Geometric Science of Information, Lecture Notes in Computer Science,
            pages 710–717, Cham, 2021.
            https://doi.org/10.1007/978-3-030-80209-7_76.
"""

import itertools
from abc import ABC

import geomstats.backend as gs
from geomstats.geometry.spd_matrices import SPDEuclideanMetric, SPDMatrices
from geomstats.geometry.stratified.point_set import (
    Point,
    PointSet,
    PointSetMetric,
    _vectorize_point,
)
from geomstats.geometry.stratified.trees import (
    BaseTopology,
    Split,
    delete_splits,
    generate_splits,
)
from geomstats.numerics.optimizers import ScipyMinimize

# TODO: update docstrings


def make_splits(n_labels):
    """Generate all possible splits of a collection."""
    if n_labels <= 1:
        raise ValueError("`n_labels` must be greater than 1.")
    if n_labels == 2:
        yield Split(part1=[0], part2=[1])
    else:
        for split in make_splits(n_labels=n_labels - 1):
            yield Split(part1=split.part1, part2=split.part2.union((n_labels - 1,)))
            yield Split(part1=split.part1.union((n_labels - 1,)), part2=split.part2)
        yield Split(part1=list(range(n_labels - 1)), part2=[n_labels - 1])


def make_topologies(n_labels):
    """Generate all possible sets of compatible splits of a collection.

    This only works well for `len(n_labels) < 8`.
    """
    if n_labels <= 1:
        raise ValueError("The collection must have 2 elements or more.")
    if n_labels in [2, 3]:
        yield Topology(
            partition=(tuple(range(n_labels)),),
            split_sets=(list(make_splits(n_labels)),),
        )
    else:
        pendant_split = Split(part1=[n_labels - 1], part2=list(range(n_labels - 1)))
        for st in make_topologies(n_labels - 1):
            for s in st.split_sets[0]:
                new_split_set = [pendant_split]
                a, b = set(s.part1), set(s.part2)
                for t in st.split_sets[0]:
                    _, d = set(t.part1), set(t.part2)
                    if t != s:
                        # TODO: probably a bug here
                        if a.issubset(d) or b.issubset(d):
                            new_split_set.append(
                                Split(
                                    part1=t.part1, part2=t.part2.union((n_labels - 1,))
                                )
                            )
                        else:
                            new_split_set.append(
                                Split(
                                    part1=t.part2, part2=t.part1.union((n_labels - 1,))
                                )
                            )
                    else:
                        new_split_set.append(
                            Split(part1=s.part1, part2=s.part2.union((n_labels - 1,)))
                        )
                        new_split_set.append(
                            Split(part1=s.part2, part2=s.part1.union((n_labels - 1,)))
                        )
                yield Topology(
                    partition=(tuple(range(n_labels)),),
                    split_sets=(new_split_set,),
                )


def _generate_partition(n_labels, p_new):
    r"""Generate a random partition of :math:`\{0,\dots,n-1\}`.

    This algorithm works as follows: Start with a single set containing zero,
    then successively add the labels from 1 to n-1 to the partition in the
    following manner: for each label u, with probability `probability`, add the
    label u to a random existing set of the partition, else add a new singleton
    set {u} to the partition (i.e. with probability 1 - `probability`).

    Parameters
    ----------
    p_new : float
        A float between 0 and 1, the probability that no new component is added,
        and 1 - probability that a new component is added.

    Returns
    -------
    partition : list[list[int]]
        A partition of the set :math:`\{0,\dots,n-1\}` into non-empty sets.
    """
    _partition = [[0]]
    for u in range(1, n_labels):
        if gs.random.rand(1) < p_new:
            index = int(gs.random.randint(0, len(_partition), (1,)))
            _partition[index].append(u)
        else:
            _partition.append([u])
    return _partition


def generate_random_wald(n_labels, p_keep, p_new, btol=1e-8, check=True):
    """Generate a random instance of class ``Wald``.

    Parameters
    ----------
    n_labels : int
        The number of labels the wald is generated with respect to.
    p_keep : float
        The probability will be inserted into the generation of a partition as
        well as for the generation of a split set for the topology of the wald.
    p_new : float
        A float between 0 and 1, the probability that no new component is added,
        and probability of 1 - p_new_ that a new component is added.
    btol: float
        Tolerance for the boundary of the coordinates in each grove. Defaults to
        1e-08.
    check : bool
        If True, checks if splits still separate all labels. In this case, the split
        will not be deleted. If False, any split can be randomly deleted.

    Returns
    -------
    random_wald : Wald
        The randomly generated wald.
    """
    partition = _generate_partition(n_labels=n_labels, p_new=p_new)
    split_sets = [generate_splits(labels=_part) for _part in partition]

    split_sets = [
        delete_splits(splits=splits, labels=part, p_keep=p_keep, check=check)
        for part, splits in zip(partition, split_sets)
    ]

    topology = Topology(partition=partition, split_sets=split_sets)
    weights = gs.random.uniform(
        size=(len(topology.flatten(split_sets)),), low=0, high=1
    )
    weights = gs.minimum(gs.maximum(btol, weights), 1 - btol)
    return Wald(topology=topology, weights=weights)


class Topology(BaseTopology):
    r"""The topology of a forest, using a split-based graph-structure representation.

    Parameters
    ----------
    partition : tuple
        A tuple of tuples that is a partition of the set :math:`\{0,\dots,n-1\}`,
        representing the label sets of each connected component of the forest topology.
    split_sets : tuple
        A tuple of tuples containing splits, where each set of splits contains only
        splits of the respective label set in the partition, so their order
        is related. The splits are the edges of the connected components of the forest,
        respectively, and thus the union of all sets of splits yields all edges of the
        forest topology.

    Attributes
    ----------
    n_labels : int
        Number of labels, the set of labels is then :math:`\{0,\dots,n-1\}`.
    where : dict
        Give the index of a split in the flattened list of all splits.
    sep : list of int
        An increasing list of numbers between 0 and m, where m is the total number
        of splits in ``self.split_sets``, starting with 0, where each number
        indicates that a new connected component starts at that index.
        Useful for example for unraveling the tuple of all splits into
        ``self.split_sets``.
    paths : list of dict
        A list of dictionaries, each dictionary is for the respective connected
        component of the forest, and the items of each dictionary are for each pair
        of labels u, v, u < v in the respective component, a list of the splits on the
        unique path between the labels u and v.
    support : list of array-like
        For each split, give an :math:`n\times n` dimensional matrix, where the
        uv-th entry is ``True`` if the split separates the labels u and v, else
        ``False``.
    """

    def __init__(self, partition, split_sets):
        self._check_init(partition, split_sets)

        super().__init__()

        self.n_labels = len(set.union(*[set(part) for part in partition]))

        partition = [tuple(sorted(part)) for part in partition]
        seq = [part[0] for part in partition]
        sort_key = sorted(range(len(seq)), key=seq.__getitem__)

        self.partition = tuple([partition[key] for key in sort_key])
        self.split_sets = tuple([tuple(sorted(split_sets[key])) for key in sort_key])

        self.where = {
            split_set: i for i, split_set in enumerate(self.flatten(self.split_sets))
        }

        lengths = [len(splits) for splits in self.split_sets]
        self.sep = [0] + [sum(lengths[0:j]) for j in range(1, len(lengths) + 1)]

        self.paths = [
            {
                (u, v): [s for s in splits if s.separates(u, v)]
                for u, v in itertools.combinations(part, r=2)
            }
            for part, splits in zip(self.partition, self.split_sets)
        ]

        support = [
            gs.zeros((self.n_labels, self.n_labels), dtype=int)
            for _ in self.flatten(self.split_sets)
        ]
        for path_dict in self.paths:
            for (u, v), path in path_dict.items():
                for split in path:
                    support[self.where[split]][u][v] = True
                    support[self.where[split]][v][u] = True
        self.support = gs.reshape(
            gs.array([m for m in self.flatten(support)]),
            (-1, self.n_labels, self.n_labels),
        )

    def _check_init(self, partition, split_sets):
        if len(split_sets) != len(partition):
            raise ValueError(
                "Number of split sets is not equal to number of components."
            )

        for part, splits in zip(partition, split_sets):
            for split in splits:
                if (split.part1 | split.part2) != set(part):
                    raise ValueError(
                        f"The split {split} is not a split of component {part}."
                    )

    def __eq__(self, other):
        """Check if ``self`` is equal to ``other``.

        Parameters
        ----------
        other : Topology
            The other topology.

        Returns
        -------
        is_equal : bool
            Return ``True`` if the topologies are equal, else ``False``.
        """
        equal_n = self.n_labels == other.n_labels
        equal_partition = self.partition == other.partition
        equal_split_sets = self.split_sets == other.split_sets
        return equal_n and equal_partition and equal_split_sets

    def __hash__(self):
        """Compute the hash of a topology.

        Note that this hash simply uses the hash function for tuples.

        Returns
        -------
        hash_of_topology : int
            Return the hash of the topology.
        """
        return hash((self.n_labels, self.partition, self.split_sets))

    def __le__(self, other):
        """Check if ``self`` is less than or equal to ``other``.

        This partial ordering is the one defined in [1] and to show if self <= other is
        True, three things must be satisfied.
        (i)     ``self.partition`` must be a refinement of ``other.partition`` in the
                sense of partitions.
        (ii)    The splits of each component in ``self`` must be contained in the
                set of splits of ``other`` restricted to the component of ``self``.
        (iii)   Whenever two components of ``self`` are contained in a component of
                ``other``, there needs to exist a split in ``other`` separating those
                two components.
        If one of those three conditions are not fulfilled, this method returns False.

        Parameters
        ----------
        other : Topology
            The structure to which self is compared to.

        Returns
        -------
        is_less_than_or_equal : bool
            Return ``True`` if (i), (ii) and (iii) are satisfied, else ``False``.
        """

        class NotPartialOrder(Exception):
            """Raise an exception when less equal is not true."""

        x_parts = [set(x) for x in self.partition]
        y_parts = [set(y) for y in other.partition]
        # (i)
        try:
            cover = {
                i: [j for j, y in enumerate(y_parts) if x.issubset(y)][0]
                for i, x in enumerate(x_parts)
            }
        except IndexError:
            return False
        # (ii)
        try:
            for (i, j), x in zip(cover.items(), x_parts):
                y_splits_restricted = {
                    split_y.restrict_to(subset=x) for split_y in other.split_sets[j]
                }
                if not set(self.split_sets[i]).issubset(y_splits_restricted):
                    raise NotPartialOrder()
        except NotPartialOrder:
            return False
        # (iii)
        try:
            for j in range(len(y_parts)):
                xs_in_y = [x for i, x in enumerate(x_parts) if cover[i] == j]
                for x1, x2 in itertools.combinations(xs_in_y, r=2):
                    sep_sp = [sp for sp in other.split_sets[j] if sp.separates(x1, x2)]
                    if not sep_sp:
                        raise NotPartialOrder()
        except NotPartialOrder:
            return False
        return True

    def __repr__(self):
        """Return the string representation of the topology.

        This string representation requires that one can retrieve all necessary
        information from the string.

        Returns
        -------
        string_of_topology : str
            Return the string representation of the topology.
        """
        return str((self.n_labels, self.partition, self.split_sets))

    def __str__(self):
        """Return the fancy printable string representation of the topology.

        This string representation does NOT require that one can retrieve all necessary
        information from the string, but this string representation is required to be
        readable by a human.

        Returns
        -------
        string_of_topology : str
            Return the fancy readable string representation of the topology.
        """
        comps = [", ".join(str(sp) for sp in splits) for splits in self.split_sets]
        return "(" + "; ".join(comps) + ")"

    @property
    def n_splits(self):
        return gs.sum(a=[len(splits) for splits in self.split_sets], dtype=int)

    def corr(self, weights):
        """Compute the correlation matrix of the topology with given edge weights.

        Parameters
        ----------
        weights : array-like, shape=[n_splits]
            Edge weights.

        Returns
        -------
        corr : array-like, shape=[n, n]
            Returns the corresponding correlation matrix.
        """
        corr = gs.zeros((self.n_labels, self.n_labels))
        for path_dict in self.paths:
            for (u, v), path in path_dict.items():
                corr[u][v] = gs.prod([1 - weights[self.where[split]] for split in path])
                corr[v][u] = corr[u][v]

        corr = gs.array(corr)
        return corr + gs.eye(corr.shape[0])

    def corr_gradient(self, weights):
        """Compute the gradient of the correlation matrix differentiated by weights.

        Parameters
        ----------
        weights : array-like, shape=[n_splits]
            The vector weights at which the gradient is computed.

        Returns
        -------
        gradient : array-like, shape=[n_splits, n, n]
            The gradient of the correlation matrix, differentiated by weights.
        """
        weights_list = [
            [y if i != k else 0 for i, y in enumerate(weights)]
            for k in range(len(weights))
        ]
        gradient = gs.array(
            [
                -supp * self.corr(weights_)
                for supp, weights_ in zip(self.support, weights_list)
            ]
        )
        return gradient

    def unflatten(self, x):
        """Transform list into list of lists according to separators ``self.sep``.

        The separators are a list of integers, increasing. Then, all elements between to
        indices in separators will be put into a list, and together, all lists give a
        nested list.

        Parameters
        ----------
        x : iterable
            The flat list that will be nested.

        Returns
        -------
        x_nested : list[list]
            The nested list of lists.
        """
        # TODO: check if it is really required
        return [x[i:j] for i, j in zip(self.sep[:-1], self.sep[1:])]

    @staticmethod
    def flatten(x):
        """Flatten a list of lists into a single list by concatenation.

        Parameters
        ----------
        x : nested list
            The nested list to flatten.

        Returns
        -------
        x_flat : list, tuple
            The flatted list.
        """
        # TODO: check if it is really required
        return [y for z in x for y in z]


class Wald(Point):
    r"""A class for wälder, that are phylogenetic forests, elements of the Wald Space.

    Parameters
    ----------
    topology : Topology
        The structure of the forest.
    weights : array-like
        The edge weights, array of floats between 0 and 1, with m entries, where m is
        the total number of splits/edges in the structure ``top``.
    """

    def __init__(self, topology, weights):
        super().__init__()
        self.topology = topology
        self.weights = weights
        # TODO: do we need to compute it?
        self.corr = self.topology.corr(weights)

    @property
    def n_labels(self):
        """Get number of labels."""
        # TODO: n_labels should be on the tree
        return self.topology.n_labels

    def __eq__(self, other):
        """Check for equal hashes of the two wälder.

        Parameters
        ----------
        other : Wald
            The other wald.

        Returns
        -------
        is_equal : bool
            Return ``True`` if the wälder are equal, else ``False``.
        """
        return hash(self) == hash(other)

    def __hash__(self):
        """Compute the hash of the wald.

        Note that this hash simply uses the hash function for tuples.

        Returns
        -------
        hash_of_wald : int
            Return the hash of the wald.
        """
        return hash((self.topology, tuple(self.weights)))

    def __repr__(self):
        """Return the string representation of the wald.

        This string representation requires that one can retrieve all necessary
        information from the string.

        Returns
        -------
        string_of_wald : str
            Return the string representation of the wald.
        """
        return repr((self.topology, tuple(self.weights)))

    def __str__(self):
        """Return the fancy printable string representation of the wald.

        This string representation does NOT require that one can retrieve all necessary
        information from the string, but this string representation is required to be
        readable by a human.

        Returns
        -------
        string_of_wald : str
            Return the fancy readable string representation of the wald.
        """
        return f"({str(self.topology)};{str(self.weights)})"

    def to_array(self):
        """Turn the wald into a numpy array, namely its correlation matrix.

        Returns
        -------
        array_of_wald : array-like, shape=[n, n]
            The correlation matrix corresponding to the wald.
        """
        return self.corr


class WaldSpace(PointSet):
    """Class for the Wald space, a metric space for phylogenetic forests.

    Parameters
    ----------
    n_labels : int
        Integer determining the number of labels in the forests, and thus the shape of
        the correlation matrices: n_labels x n_labels.

    Attributes
    ----------
    ambient_space : Manifold
        The ambient space, the positive definite n_labels x n_labels matrices that the
        WaldSpace is embedded into.
    """

    def __init__(self, n_labels, equip=True):
        super().__init__(equip=equip)
        self.n_labels = n_labels

        self.ambient_space = SPDMatrices(n=self.n_labels, equip=False)
        # TODO: pass it as input?
        self.projection_solver = LocalProjectionSolver()

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return WaldSpaceMetric

    @property
    def stratum_metric(self):
        return self.ambient_space.metric

    @_vectorize_point((1, "point"))
    def belongs(self, point):
        """Check if a point `wald` belongs to Wald space.

        From FUTURE PUBLICATION we know that the corresponding matrix of a wald is
        strictly positive definite if and only if the labels are separated by at least
        one edge, which is exactly when the wald is an element of the Wald space.

        Parameters
        ----------
        point : Wald or list of Wald
            The point to be checked.

        Returns
        -------
        belongs : bool
            Boolean denoting if `point` belongs to Wald space.
        """
        is_spd = [
            self.ambient_space.belongs(single_point.to_array())
            for single_point in point
        ]
        is_between_0_1 = [
            gs.all(w.weights > 0) and gs.all(w.weights < 1) for w in point
        ]
        results = [is1 and is2 for is1, is2 in zip(is_spd, is_between_0_1)]
        return results

    def random_point(self, n_samples=1, p_tree=0.9, p_keep=0.9, btol=1e-8):
        """Sample a random point in Wald space.

        Parameters
        ----------
        n_samples : int
            Number of samples. Defaults to 1.
        p_tree : float between 0 and 1
            The probability that the sampled point is a tree, and not a forest. If the
            probability is equal to 1, then the sampled point will be a tree.
            Defaults to 0.9.
        p_keep : float between 0 and 1
            The probability that a sampled edge is kept and not deleted randomly.
            To be precise, it is not exactly the probability, as some edges cannot be
            deleted since the requirement that two labels are separated by a split might
            be violated otherwise.
            Defaults to 0.9
        btol: float
            Tolerance for the boundary of the coordinates in each grove. Defaults to
            1e-08.

        Returns
        -------
        samples : Wald or list of Wald, shape=[n_samples]
            Points sampled in Wald space.
        """
        p_new = p_tree ** (1 / (self.n_labels - 1))
        forests = [
            generate_random_wald(self.n_labels, p_keep, p_new, btol, check=True)
            for _ in range(n_samples)
        ]

        if n_samples == 1:
            return forests[0]

        return forests

    @_vectorize_point((1, "point"))
    def set_to_array(self, points):
        """Convert a set of points into an array.

        Parameters
        ----------
        points : list of Wald, shape=[...]
            Number of samples of wälder to turn into an array.

        Returns
        -------
        points_array : array-like, shape=[...]
            Array of the wälder that are turned into arrays.
        """
        results = gs.array([wald.to_array() for wald in points])
        return results

    def lift(self, point):
        """Lift a point to the ambient space.

        Returns
        -------
        ambient_point : array-like, shape=[..., n_labels, n_labels]
        """
        # TODO: handle vectorization
        # TODO: is extend a better name?
        return point.corr

    def projection(self, ambient_point, **kwargs):
        """Projects a point into Wald space."""
        return self.projection_solver.projection(self, ambient_point, **kwargs)


class WaldSpaceMetric(PointSetMetric):
    # TODO: delete

    # geodesic algorithms

    # naive needs: s_proj, a_path

    # symmetric needs: s_proj, a_path_t

    # s_proj needs _proj_target_gradient (changes with ambient metric)
    # _proj_target_gradient needs s_chart_and_gradient
    # s_chart and s_chart_gradient available in ftools and do not
    # depend in ambient metric

    # straightning-ext needs: a_log, a_exp, s_proj, starting path (e.g. naive)

    def squared_dist(self, point_a, point_b):
        """Compute the squared distance between two points.

        Parameters
        ----------
        point_a : Wald or list[Wald]
            A point in wald space.
        point_b : Wald or list[Wald]
            A point in BHV Space.

        Returns
        -------
        squared_dist : float or gs.array
            The squared distance between the two points.
        """
        # TODO: handle vectorization

        ambient_point_a = self._space.lift(point_a)
        ambient_point_b = self._space.lift(point_b)

        return self._space.stratum_metric.squared_dist(ambient_point_a, ambient_point_b)

    def dist(self, point_a, point_b):
        """Compute the distance between two points.

        Parameters
        ----------
        point_a : Wald or list[Wald]
            A point in wald space.
        point_b : Wald or list[Wald]
            A point in wald space.

        Returns
        -------
        dist : float
            The distance between the two points.
        """
        return gs.sqrt(self.squared_dist(point_a, point_b))

    def geodesic(self):
        pass


class _BaseProjectionSolver(ABC):
    # TODO: make it more complete
    def __init__(self):
        # TODO: move this to metric?
        # TODO: need to see how connected this is to the solver
        self._map_ambient_metric_to_target_gradient = {
            SPDEuclideanMetric: self._euclidean_target_gradient,
        }

    def _euclidean_target_gradient(self, weights, topology, ambient_point, space):
        corr = topology.corr(weights)
        grad = topology.corr_gradient(weights)

        target = space.stratum_metric.squared_dist(corr, ambient_point)
        target_grad = gs.array(
            [2 * gs.sum((corr - ambient_point) * grad_) for grad_ in grad]
        )

        return target, target_grad

    def _proj_target_gradient(self, space, ambient_point, topology):
        metric_target_gradient = self._map_ambient_metric_to_target_gradient[
            type(space.stratum_metric)
        ]

        return lambda x: metric_target_gradient(
            weights=x,
            topology=topology,
            ambient_point=ambient_point,
            space=space,
        )


class LocalProjectionSolver(_BaseProjectionSolver):
    # TODO: may need to handle geodesic due to different inputs?

    # BUG: projection gets fully ones sometimes (Eucludian)

    def __init__(self, btol=1e-10):
        super().__init__()
        self.btol = btol

        self.optimizer = ScipyMinimize(
            method="L-BFGS-B",
            jac=True,
            options=dict(gtol=1e-5, ftol=2.22e-9),
        )

    def _get_bounds(self, n_splits):
        return [(self.btol, 1 - self.btol)] * n_splits

    def projection(self, space, ambient_point, topology):
        if len(topology.partition) == topology.n_labels:
            return Wald(topology=topology, weights=gs.ones(topology.n_labels))

        target_and_gradient = self._proj_target_gradient(
            space=space,
            ambient_point=ambient_point,
            topology=topology,
        )

        n_splits = topology.n_splits
        self.optimizer.bounds = self._get_bounds(n_splits)

        x0 = gs.ones(n_splits) * 0.5

        res = self.optimizer.minimize(
            target_and_gradient,
            x0,
        )

        if res.status != 0:
            raise ValueError("Projection failed!")

        x = [
            _x if self.btol < _x < 1 - self.btol else 0 if _x <= self.btol else 1
            for _x in res.x
        ]

        return Wald(topology=topology, weights=x)

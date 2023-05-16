"""Class for the BHV Tree Space.

Class ``Tree``.
A tree is essentially a phylogenetic tree with edges having length greater than zero.
The representation of the tree is via splits, the edge lengths are stored in a vector.

Class ``TreeSpace``.
A topological space. Points in Tree space are instances of the class :class:`Tree`:
phylogenetic trees with edge lengths between 0 and infinity.
For the space of trees see also [BHV01].

Class ``BHVSpace``.
The BHV Tree Space as it is introduced in [BHV01], a metric space that is CAT(0), and
there exist unique geodesics between each pair of points in the BHV Space.
The polynomial time algorithm for computing the distance and geodesic between two points
is implemented, following the definitions and results of [OP11].
There, computing the geodesic between two trees is called the 'Geodesic Tree Path'
problem, and that is why some methods below (not visible to the user though) start with
the letters 'gtp'.

Lead author: Jonas Lueg

References
----------
[BHV01] Billera, L. J., S. P. Holmes, K. Vogtmann.
        "Geometry of the Space of Phylogenetic Trees."
        Advances in Applied Mathematics,
        volume 27, issue 4, pages 733-767, 2001.
        https://doi.org/10.1006%2Faama.2001.0759

[OP11]  Owen, M., J. S. Provan.
        "A Fast Algorithm for Computing Geodesic Distances in Tree Space."
        IEEE/ACM Transactions on Computational Biology and Bioinformatics,
        volume 8, issue 1, pages 2-13, 2011.
        https://doi.org/10.1109/TCBB.2010.3
"""
import itertools

import networkx as nx

# TODO: only needed for np.inf
import numpy as np

import geomstats.backend as gs
from geomstats.geometry.stratified.point_set import (
    Point,
    PointSet,
    PointSetMetric,
    _vectorize_point,
    broadcast_lists,
)
from geomstats.geometry.stratified.trees import (
    BaseTopology,
    Split,
    delete_splits,
    generate_splits,
)

# TODO: update vectorize to retrieve proper shape

# TODO: n_labels to n_leaves?


def generate_random_tree(n_labels, p_keep=0.9, btol=1e-8):
    """Generate a random instance of ``Tree``.

    Parameters
    ----------
    p_keep : float between 0 and 1
        The probability that a sampled edge is kept and not deleted randomly.
        To be precise, it is not exactly the probability, as some edges cannot be
        deleted since the requirement that two labels are separated by a split might
        be violated otherwise.
        Defaults to 0.9
    btol: float
        Tolerance for the boundary of the edge lengths. Defaults to 1e-08.
    """
    labels = list(range(n_labels))

    initial_splits = generate_splits(labels)
    splits = delete_splits(initial_splits, labels, p_keep, check=False)

    x = gs.random.uniform(size=(len(splits),), low=0, high=1)
    x = gs.minimum(gs.maximum(btol, x), 1 - btol)
    lengths = gs.maximum(btol, gs.abs(gs.log(1 - x)))

    return Tree(Topology(splits), lengths)


def get_pendant_edges(n_labels):
    """Return pendant edges.

    Parameters
    ----------
    n_labels : int

    Returns
    -------
    pendant_edges : set[Split]
        Edges containing the leaves.
    """
    return {
        Split(part1=[i], part2=[j for j in range(n_labels) if j != i])
        for i in range(n_labels)
    }


class Topology(BaseTopology):
    r"""The topology of a tree, using a split-based representation.

    Parameters
    ----------
    n_labels : int
        Number of labels, the set of labels is then :math:`\{0,\dots,n-1\}`.
    splits : list[Split]
        The structure of the tree in form of a set of splits of the set of labels.
    """

    def __init__(self, splits):
        # TODO: add verification?
        super().__init__()
        self.splits = splits
        self.splits_set = set(splits)

        self._check_init()

    def _check_init(self):
        if len(self.splits) != len(self.splits_set):
            raise ValueError("Some edges are equivalent, collapse them first.")

    @property
    def n_labels(self):
        return len(self.splits[0].part1 | self.splits[0].part2)


class Tree(Point):
    r"""A class for trees, that are phylogenetic trees, elements of the BHV space.

    Parameters
    ----------
    n_labels : int
        Number of labels, the set of labels is then :math:`\{0,\dots,n_labels-1\}`.
    splits : list[Split]
        The structure of the tree in form of a set of splits of the set of labels.
    lengths : array-like
        The edge lengths of the splits, a vector containing positive numbers.
    """

    def __init__(self, topology, lengths):
        # TODO: rename lengths to weights
        self._check_init(topology, lengths)
        super().__init__()

        self.topology = topology
        self.lengths = lengths

        self.dict_repr = dict(zip(self.topology.splits, self.lengths))

    def _check_init(self, topology, lengths):
        if len(topology.splits) != len(lengths):
            raise ValueError(
                "Number of splits does not correspond to number of lengths."
            )

    def __repr__(self):
        """Return the string representation of the tree.

        This string representation requires that one can retrieve all necessary
        information from the string.

        Returns
        -------
        string_of_tree : str
            Return the string representation of the tree.
        """
        return repr(self.dict_repr)

    def __hash__(self):
        """Compute the hash of the wald.

        Note that this hash simply uses the hash function for tuples.

        Returns
        -------
        hash_of_wald : int
            Return the hash of the wald.
        """
        return hash((self.topology, tuple(self.lengths)))

    def __str__(self):
        """Return the fancy printable string representation of the tree.

        This string representation does NOT require that one can retrieve all necessary
        information from the string, but this string representation is required to be
        readable by a human.

        Returns
        -------
        string_of_tree : str
            Return the fancy readable string representation of the tree.
        """
        return f"({self.topology};{str(self.lengths)})"

    @property
    def n_labels(self):
        return self.topology.n_labels

    def edge_length(self, split):
        return self.dict_repr[split]


class TreeSpace(PointSet):
    """Class for the Tree space, a point set containing phylogenetic trees.

    Parameters
    ----------
    n_labels : int
        The number of labels in the trees.
    splits : list[Split]
        A list of splits of the set of labels.
    """

    def __init__(self, n_labels, equip=True):
        self.n_labels = n_labels
        super().__init__(equip=equip)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return BHVMetric

    @_vectorize_point((1, "point"))
    def belongs(self, point, atol=gs.atol):
        """Check if a point belongs to Tree space.

        Parameters
        ----------
        point : Tree or list of Tree
            The point to be checked.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        belongs : bool
            Boolean denoting if `point` belongs to Tree space.
        """
        return gs.array([gs.all(tree.lengths > -atol) for tree in point]) & gs.array(
            [point_.n_labels == self.n_labels for point_ in point]
        )

    def random_point(self, n_samples=1, p_keep=0.9, btol=1e-8):
        """Sample a random point in Tree space.

        Parameters
        ----------
        n_samples : int
            Number of samples. Defaults to 1.
        p_keep : float between 0 and 1
            The probability that a sampled edge is kept and not deleted randomly.
            To be precise, it is not exactly the probability, as some edges cannot be
            deleted since the requirement that two labels are separated by a split might
            be violated otherwise.
            Defaults to 0.9
        btol: float
            Tolerance for the boundary of the edge lengths. Defaults to 1e-08.

        Returns
        -------
        samples : Tree or list of Tree, shape=[n_samples]
            Points sampled in Tree space.
        """
        trees = [
            generate_random_tree(self.n_labels, p_keep, btol) for _ in range(n_samples)
        ]

        if n_samples == 1:
            return trees[0]

        return trees


class BHVMetric(PointSetMetric):
    """BHV metric for Tree Space for phylogenetic trees.

    Parameters
    ----------
    n_labels : int
        The number of labels.
    """

    def __init__(self, space):
        super().__init__(space=space)
        # TODO: need to define n_labels
        self.geodesic_solver = GeodesicTreePathSolver(n_labels=space.n_labels)

    @property
    def n_labels(self):
        # TODO: remove?
        return self._space.n_labels

    def squared_dist(self, point_a, point_b):
        """Compute the squared distance between two points.

        Parameters
        ----------
        point_a : Tree
            A point in BHV Space.
        point_b : Tree
            A point in BHV Space.

        Returns
        -------
        squared_dist : float
            The squared distance between the two points.
        """
        return self.geodesic_solver.squared_dist(point_a, point_b)

    def dist(self, point_a, point_b):
        """Compute the distance between two points.

        Parameters
        ----------
        point_a : Tree
            A point in BHV Space.
        point_b : Tree
            A point in BHV Space.

        Returns
        -------
        dist : float
            The distance between the two points.
        """
        return self.geodesic_solver.dist(point_a, point_b)

    def geodesic(self, point_a, point_b):
        """Compute the geodesic between two points.

        Parameters
        ----------
        point_a : Tree
            A point in BHV Space.
        point_b : Tree
            A point in BHV Space.

        Returns
        -------
        geodesic : callable
            The geodesic between the two points. Takes parameter t, that is the time
            between 0 and 1 at which the corresponding point on the path is returned.
        """
        return self.geodesic_solver.geodesic(point_a=point_a, point_b=point_b)


class GeodesicTreePathSolver:
    """'Geodesic Tree Path' problem solver [OP11]_.

    Essentially uses Theorem 2.4 from [OP11]_.

    Parameters
    ----------
    tol : float
        Tolerance for the algorithm, in particular for the decision problem in the
        GTP algorithm in [OP11] to avoid unambiguity.
    """

    # TODO: bring in notion of path (as sequence of orthants)?

    def __init__(self, n_labels, tol=1e-8):
        # TODO: do we really need n_labels or receive space??
        self.n_labels = n_labels
        self.tol = tol

    def _squared_dist_single(self, point_a, point_b):
        common_splits_with_length, supports = self._trees_with_common_support(
            point_a,
            point_b,
        )
        sq_dist_common = sum(
            (length[0] - length[1]) ** 2
            for length in common_splits_with_length.values()
        )
        sq_dist_parts = sum(
            (
                gs.sqrt(sum(point_a.edge_length(split) ** 2 for split in splits_a))
                + gs.sqrt(sum(point_b.edge_length(split) ** 2 for split in splits_b))
            )
            ** 2
            for support_a, support_b in supports.values()
            for splits_a, splits_b in zip(support_a, support_b)
        )

        return sq_dist_common + sq_dist_parts

    @_vectorize_point((1, "point_a"), (2, "point_b"))
    def squared_dist(self, point_a, point_b):
        """Compute the squared distance between two points.

        Parameters
        ----------
        point_a : Tree or list[Tree]
            A point in BHV Space.
        point_b : Tree or list[Tree]
            A point in BHV Space.

        Returns
        -------
        squared_dist : float or gs.array
            The squared distance between the two points.
        """
        # TODO: make this come from decorator
        point_a, point_b = broadcast_lists(point_a, point_b)

        sq_dists = gs.array(
            [
                self._squared_dist_single(point_a_, point_b_)
                for point_a_, point_b_ in zip(point_a, point_b)
            ]
        )

        if len(sq_dists) == 1:
            return sq_dists[0]

        return sq_dists

    def dist(self, point_a, point_b):
        """Compute the distance between two points.

        Parameters
        ----------
        point_a : Tree
            A point in BHV Space.
        point_b : Tree
            A point in BHV Space.

        Returns
        -------
        dist : float
            The distance between the two points.
        """
        return gs.sqrt(self.squared_dist(point_a, point_b))

    def _geodesic_single(self, point_a, point_b):
        """Compute the geodesic between two points.

        Parameters
        ----------
        point_a : Tree
            A point in BHV Space.
        point_b : Tree
            A point in BHV Space.

        Returns
        -------
        geodesic : callable
            The geodesic between the two points. Takes parameter t, that is the time
            between 0 and 1 at which the corresponding point on the path is returned.
        """
        # TODO: delete
        sp_a = dict(zip(point_a.topology.splits, point_a.lengths))
        sp_b = dict(zip(point_b.topology.splits, point_b.lengths))
        common_a, common_b, supports = self._trees_with_common_support(
            sp_a,
            sp_b,
        )

        common_splits_with_length, supports = self._trees_with_common_support(
            point_a,
            point_b,
        )

        # TODO: return topology of crossed orthant?

        # TODO: remove partitions
        ratios = {
            partition: [
                gs.sqrt(
                    sum(point_a.edge_length(split) ** 2 for split in splits_a)
                    / sum(point_b.edge_length(split) ** 2 for split in splits_b)
                )
                for splits_a, splits_b in zip(support_a, support_b)
            ]
            for partition, (support_a, support_b) in supports.items()
        }

        def geodesic_t(t):
            if t == 0.0:
                return point_a
            elif t == 1.0:
                return point_b

            t_ratio = t / (1 - t)
            splits_t = {s: (1 - t) * common_a[s] + t * common_b[s] for s in common_a}
            for part, (supp_a, supp_b) in supports.items():
                index = gs.argmax([t_ratio <= _r for _r in ratios[part] + [np.inf]])
                splits_t_a = {
                    s: sp_a[s] * (1 - t - t / _r)
                    for a_k, _r in zip(supp_a[index:], ratios[part][index:])
                    for s in a_k
                }
                splits_t_b = {
                    s: sp_b[s] * (t - (1 - t) * _r)
                    for b_k, _r in zip(supp_b[:index], ratios[part][:index])
                    for s in b_k
                }
                splits_t = {**splits_t, **splits_t_a, **splits_t_b}

            splits_lengths = [
                (split, length)
                for split, length in splits_t.items()
                if length > self.tol
            ]
            tree_t = Tree(
                n_labels=self.n_labels,
                splits=[sl[0] for sl in splits_lengths],
                lengths=[sl[1] for sl in splits_lengths],
            )
            return tree_t

        def geodesic_(t):
            if isinstance(t, (float, int)):
                t = gs.array([t])

            return [geodesic_t(t_) for t_ in t]

        return geodesic_

    @_vectorize_point((1, "point_a"), (2, "point_b"))
    def geodesic(self, point_a, point_b):
        """Compute the geodesic between two points.

        Parameters
        ----------
        point_a : Tree or list[Tree]
            A point in BHV Space.
        point_b : Tree or list[Tree]
            A point in BHV Space.

        Returns
        -------
        geodesic : callable
            The geodesic between the two points. Takes parameter t, that is the time
            between 0 and 1 at which the corresponding point on the path is returned.
        """

        # TODO: generalize; also used in spider?
        def _vec(t, fncs):
            if len(fncs) == 1:
                return fncs[0](t)

            return [fnc(t) for fnc in fncs]

        point_a, point_b = broadcast_lists(point_a, point_b)

        fncs = [
            self._geodesic_single(point_a_, point_b_)
            for point_a_, point_b_ in zip(point_a, point_b)
        ]

        return lambda t: _vec(t, fncs=fncs)

    def _trees_with_common_support(self, point_a, point_b):
        """Compute the support that corresponds to a geodesic for common split sets.

        We refer to the splits of each tree ``A`` and ``B``, respectively.
        This method divides the split sets into smaller split sets that have distinct
        support and then use the method ``gtp_trees_with_distinct_support``.
        For each of these smaller subsets, return the support in a dictionary.

        The common splits are returned together and with the respective edge
        lengths. A split in ``A`` that is not in ``B`` but compatible with all
        splits of ``B`` is added to the common splits of ``B`` with length zero,
        and vice versa for splits in ``B``.

        Parameters
        ----------
        point_a : Tree
            A point in BHV Space.
        point_b : Tree
            A point in BHV Space.

        Returns
        -------
        common_splits_with_length : dict[tuple[2, float]]
            Contains the splits of ``A`` that are also in ``B``, the splits of ``B``
            that are compatible with all splits in ``A``, given edge length zero,
            and the splits of ``A`` that are compatible with all splits in ``B``,
            given edge length zero.
        supports: dict
            Contains the respective support for each subtree.
        """
        pendants = get_pendant_edges(self.n_labels)

        common_splits = point_a.topology.splits_set & point_b.topology.splits_set

        # only for degenerate trees
        splits_only_a = point_a.topology.splits_set - common_splits
        splits_only_b = point_b.topology.splits_set - common_splits
        easy_a = {
            split
            for split in splits_only_a
            if split.is_compatible_with_set(splits_only_b)
        }
        easy_b = {
            split
            for split in splits_only_b
            if split.is_compatible_with_set(splits_only_a)
        }

        total_a = (point_a.topology.splits_set | easy_b) - pendants
        total_b = (point_b.topology.splits_set | easy_a) - pendants

        cut_splits = tuple((common_splits | easy_a | easy_b) - pendants)

        partitions_a = self._cut_tree_at_splits(total_a, cut_splits)
        partitions_b = self._cut_tree_at_splits(total_b, cut_splits)
        supports = {
            partition: self._trees_with_distinct_support(
                {split: point_a.dict_repr[split] for split in partitions_a[partition]},
                {split: point_b.dict_repr[split] for split in partitions_b[partition]},
            )
            for partition in partitions_a.keys()
            if partitions_a[partition] and partitions_b[partition]
        }

        common_splits = common_splits | easy_a | easy_b
        common_splits_with_length = {
            split: (
                point_a.dict_repr.get(split, 0.0),
                point_b.dict_repr.get(split, 0.0),
            )
            for split in common_splits
        }
        return common_splits_with_length, supports

    def _cut_tree_at_splits(self, splits, cut_splits):
        """Cut a tree, given by splits, at all edges in cut_splits.

        Starting with the partition that consists of all labels and is assigned all
        splits,
        the tree is successively cut into parts by the splits in cut_splits.
        Accordingly, the set of labels is cut successively into parts and the set of all
        splits is also cut successively into the respective parts.

        Parameters
        ----------
        splits : iterable of split
            The tree given via its splits. Each split corresponds to an edge.
        cut_splits : iterable of Split
            A subset of splits, the edges at which the tree is cut.

        Returns
        -------
        partition : dict of tuple, tuple
            A dictionary, where the keys form a partition of the set of labels
            (0,...,n_labels-1),
            and each key is assigned the tuple of splits that are part of the subtree
            that the respective set of labels is spanning.
        """
        partition = {tuple(range(self.n_labels)): splits}
        for cut in cut_splits:
            for labels, subtree in partition.items():
                if cut in subtree:
                    break
            else:
                continue

            splits = subtree - {cut}
            part1 = set(labels) & set(cut.part1)
            part2 = set(labels) & set(cut.part2)
            subtree1 = {
                split for split in splits if part1 == cut.get_part_towards(split)
            }
            subtree2 = splits - subtree1

            partition.pop(labels)
            partition = {
                **partition,
                tuple(part1): subtree1,
                tuple(part2): subtree2,
            }
        return partition

    def _trees_with_distinct_support(self, splits_a, splits_b):
        """Compute the support that corresponds to a geodesic for disjoint split sets.

        This is essentially the GTP algorithm from [1], starting with a cone path and
        iteratively updating the support, solving in each iteration an extension problem
        for
        each support pair.

        The Extension Problem gives a minimum cut of a graph and two-set partitions C1
        and
        C2 of A, and D1 and D2 of B, respectively. If the value of the minimum cut is
        greater or equal to one minus some tolerance, then the support pair (A,B) is
        split
        into (C1,D1) and (C2,D2).

        Parameters
        ----------
        splits_a : dict of Split, float
            The splits in A and their respective lengths.
        splits_b : dict of Split, float
            The splits in B and their respective lengths.

        Returns
        -------
        support_a : tuple of tuple
            The support partition of A corresponding to a geodesic.
        support_b : tuple of tuple
            The support partition of B corresponding to a geodesic.
        """
        old_support_a = (tuple(splits_a.keys()),)
        old_support_b = (tuple(splits_b.keys()),)
        weights_a = {split: splits_a[split] ** 2 for split in splits_a}
        weights_b = {split: splits_b[split] ** 2 for split in splits_b}
        while True:
            new_support_a, new_support_b = tuple(), tuple()
            for pair_a, pair_b in zip(old_support_a, old_support_b):
                pair_a_w = {split: weights_a[split] for split in pair_a}
                pair_b_w = {split: weights_b[split] for split in pair_b}
                value, c1, c2, d1, d2 = self._solve_extension_problem(
                    pair_a_w, pair_b_w
                )

                if value >= 1 - self.tol:
                    new_support_a += (pair_a,)
                    new_support_b += (pair_b,)
                else:
                    new_support_a += (c1, c2)
                    new_support_b += (d1, d2)

            if len(new_support_a) == len(old_support_a):
                break

            old_support_a, old_support_b = new_support_a, new_support_b

        return new_support_a, new_support_b

    @staticmethod
    def _construct_incompatibility_graph(sq_splits_a, sq_splits_b):
        total_a, total_b = sum(sq_splits_a.values()), sum(sq_splits_b.values())

        graph = nx.DiGraph()

        for split, weight in sq_splits_a.items():
            graph.add_edge("source", split, capacity=weight / total_a)

        for split, weight in sq_splits_b.items():
            graph.add_edge(split, "sink", capacity=weight / total_b)

        for split_a, split_b in itertools.product(
            sq_splits_a.keys(), sq_splits_b.keys()
        ):
            if not split_a.is_compatible(split_b):
                graph.add_edge(split_a, split_b)

        return graph

    def _solve_extension_problem(self, sq_splits_a, sq_splits_b):
        """Solve the extension problem in [1] for sets of splits with squared weights.

        Solving the min weight vertex cover with respect to the incompatibility graph in
        the Extension Problem in [1] is equivalent to solving the minimum cut problem
        for
        the following directed graph with edges that have 'capacities'.
        The set of vertices are the splits in A, the splits in B, a sink and a source
        node.
        The source is connected to all splits in A, each edge has the normalized squared
        weight of the split it is attached to. Analogously, each split in B is connected
         to
        the sink and the corresponding edge has normalized squared weight of the split
        in B.
        Finally, each split in A is attached to a split in B whenever the splits are not
        compatible. The edge is given infinite capacity.

        The minimum cut returns the two-set partition (V, V_bar) of the set of vertices
        and
        its value, that is the sum of all capacities of edges from V to V_bar, such
        that the
        source is in V and the sink is in V_bar.

        If the value is larger or equal than one (possibly with respect to some
        tolerance),
        then a geodesic is found and there is no need to update anything.
        Else, the sets A and B are separated into sets
        C_1 = A intersection V_bar, C_2 = A intersection V,
        D_1 = B intersection V_bar, D_2 = B intersection V.
        Then, the new support is (i.e. A and B are replaced with) (C_1, C_2) and
        (D_1, D_2)
        (here, the notation from [1], GTP algorithm is used).

        Parameters
        ----------
        sq_splits_a : dict of Split, float
            Dictionary of splits in A with squared length associated to each split.
        sq_splits_b : dict of Split, float
            Dictionary of splits in B with squared length associated to each split.

        Returns
        -------
        value : float
            The value of the minimum cut.
        c1 : set of Split
            First part of A that it is split into.
        c2 : set of Split
            Second part of A that it is split into.
        d1 : set of Split
            First part of B that it is split into.
        d2 : set of Split
            Second part of B that it is split into.
        """
        graph = self._construct_incompatibility_graph(sq_splits_a, sq_splits_b)

        min_value, (v, v_bar) = nx.minimum_cut(graph, "source", "sink")

        a = set(sq_splits_a.keys())
        b = set(sq_splits_b.keys())

        v = set(v)
        v_bar = set(v_bar)

        c2 = tuple(a & v)

        # TODO: why isn't the problem symmetric in some cases?
        # this is just a patch to guarantee symmetric distance
        if min_value < 1 - self.tol and not c2:
            graph = self._construct_incompatibility_graph(sq_splits_b, sq_splits_a)

            min_value, (v_bar, v) = nx.minimum_cut(graph, "source", "sink")
            c2 = tuple(a & v)

        return min_value, tuple(a & v_bar), c2, tuple(b & v_bar), tuple(b & v)

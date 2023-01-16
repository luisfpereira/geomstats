import geomstats.backend as gs
from geomstats.geometry.base import LevelSet, VectorSpace
from geomstats.geometry.lie_group import LieGroup
from geomstats.test.vectorization import get_n_points


def _get_random_tangent_vec_vector_space(space, base_point):
    n_points = get_n_points(space, base_point)
    vec = space.random_point(n_points)
    return space.to_tangent(vec, base_point)


def _get_random_tangent_vec_lie_group(space, base_point):
    n_points = get_n_points(space, base_point)
    batch_shape = (n_points,) if n_points > 1 else ()
    vec = gs.random.normal(size=batch_shape + space.shape)
    return space.to_tangent(vec, base_point)


def _get_random_tangent_vec_level_set(space, base_point):
    n_points = get_n_points(space, base_point)
    vec = space.embedding_space.random_point(n_points)
    return space.to_tangent(vec, base_point)


def get_random_tangent_vec(space, base_point):
    if isinstance(space, LevelSet):
        return _get_random_tangent_vec_level_set(space, base_point)

    if isinstance(space, LieGroup):
        return _get_random_tangent_vec_lie_group(space, base_point)

    if isinstance(space, VectorSpace):
        return _get_random_tangent_vec_vector_space(space, base_point)


def get_random_quaternion(n_points=1):
    # https://stackoverflow.com/a/44031492/11011913
    size = (3, n_points) if n_points > 1 else 3
    u, v, w = gs.random.uniform(size=size)

    return gs.transpose(
        gs.array(
            [
                gs.sqrt(1 - u) * gs.sin(2 * gs.pi * v),
                gs.sqrt(1 - u) * gs.cos(2 * gs.pi * v),
                gs.sqrt(u) * gs.sin(2 * gs.pi * w),
                gs.sqrt(u) * gs.cos(2 * gs.pi * w),
            ]
        )
    )
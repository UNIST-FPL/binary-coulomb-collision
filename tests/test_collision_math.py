import numpy as np
from numpy.testing import assert_allclose

from binary_collision import Collision


def test_get_h_is_orthogonal_and_preserves_relative_speed():
    g = np.array(
        [
            [3.0, 4.0, 0.0],
            [1.0, 2.0, 2.0],
            [6.0, 2.0, 3.0],
        ]
    )

    h = Collision.get_h(g, rng=7)

    assert_allclose(np.sum(g * h, axis=1), 0.0, atol=1.0e-12)
    assert_allclose(np.linalg.norm(h, axis=1), np.linalg.norm(g, axis=1), atol=1.0e-12)


def test_shuffle_rows_with_map_is_reversible():
    array = np.arange(12).reshape(4, 3)

    shuffled, row_map = Collision.shuffle_rows_with_map(array, rng=5)

    assert_allclose(shuffled[np.argsort(row_map)], array)


def test_get_h_handles_axis_aligned_relative_velocity():
    g = np.array(
        [
            [5.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
        ]
    )

    h = Collision.get_h(g, rng=11)

    assert_allclose(np.sum(g * h, axis=1), 0.0, atol=1.0e-12)
    assert_allclose(np.linalg.norm(h, axis=1), np.linalg.norm(g, axis=1), atol=1.0e-12)


def test_get_a_is_positive_and_scattering_cosine_stays_in_bounds(species_factory):
    _, _, _, collision = species_factory(seed=23, ion_markers=16, electron_markers=16)
    s = np.array([1.0e-8, 1.0e-4, 1.0e-2, 1.0, 10.0])

    a_values = collision.get_A(s)
    cos_ab, cos_ba = collision.evaluate_cosChi(s, s)

    assert np.all(a_values > 0.0)
    assert np.all(cos_ab <= 1.0)
    assert np.all(cos_ab >= -1.0)
    assert np.all(cos_ba <= 1.0)
    assert np.all(cos_ba >= -1.0)

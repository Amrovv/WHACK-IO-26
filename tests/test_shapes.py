import pytest

from coach.perception.shapes import (
    distance_to_polygon,
    is_valid_polygon,
    point_in_polygon,
    polygon_area,
    polygon_bounds,
    polygon_centroid,
)

SQUARE = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
# a slanted slab: a parallelogram leaning to the right as it goes down
SLAB = [(0.0, 0.0), (1.0, 0.0), (5.0, 4.0), (4.0, 4.0)]
# an L shape (concave), to check the inside test is not just the bounding box
L_SHAPE = [(0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (1.0, 1.0), (1.0, 4.0), (0.0, 4.0)]


def test_area_and_bounds():
    assert polygon_area(SQUARE) == pytest.approx(4.0)
    assert polygon_area(SLAB) == pytest.approx(4.0)  # base 1 x height 4
    assert polygon_bounds(SLAB) == (0.0, 0.0, 5.0, 4.0)
    assert polygon_area(SQUARE[::-1]) == pytest.approx(4.0)  # winding order does not matter


def test_centroid_of_square_and_parallelogram():
    assert polygon_centroid(SQUARE) == pytest.approx((1.0, 1.0))
    assert polygon_centroid(SLAB) == pytest.approx((2.5, 2.0))


def test_centroid_of_a_degenerate_polygon_falls_back_to_the_vertex_mean():
    line = [(0.0, 0.0), (2.0, 2.0), (4.0, 4.0)]
    assert polygon_centroid(line) == pytest.approx((2.0, 2.0))


def test_point_in_a_slanted_slab_is_not_just_the_bounding_box():
    assert point_in_polygon((0.6, 0.3), SLAB)  # inside the top of the slab
    assert point_in_polygon((4.4, 3.7), SLAB)  # inside the bottom of the slab
    assert not point_in_polygon((4.4, 0.3), SLAB)  # inside the box, empty wall
    assert not point_in_polygon((0.3, 3.7), SLAB)  # inside the box, empty wall


def test_point_in_a_concave_polygon():
    assert point_in_polygon((0.5, 3.0), L_SHAPE)
    assert not point_in_polygon((3.0, 3.0), L_SHAPE)  # the notch of the L


def test_distance_to_polygon_is_zero_inside_and_edge_distance_outside():
    assert distance_to_polygon((1.0, 1.0), SQUARE) == 0.0
    assert distance_to_polygon((3.0, 1.0), SQUARE) == pytest.approx(1.0)  # 1 right of an edge
    assert distance_to_polygon((3.0, 3.0), SQUARE) == pytest.approx(2**0.5)  # to a corner
    # next to the slanted edge: the distance is perpendicular to the slant, not along an axis
    assert distance_to_polygon((4.4, 0.3), SLAB) == pytest.approx(
        distance_to_polygon((4.4, 0.3), SLAB)
    )
    assert 0 < distance_to_polygon((4.4, 0.3), SLAB) < 4.4 - 1.3


def test_valid_polygon_needs_three_corners_that_are_not_in_a_line():
    assert is_valid_polygon(SQUARE) and is_valid_polygon(SLAB)
    assert not is_valid_polygon([(0.0, 0.0), (1.0, 1.0)])
    assert not is_valid_polygon([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
    assert not is_valid_polygon([])

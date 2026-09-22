import numpy as np

from rul.evolution.nsga import (
    crowding_distance,
    dominates,
    fast_non_dominated_sort,
    select_next_generation,
)


def test_dominates_strictly_better_in_one_equal_in_other():
    assert dominates((1.0, 2.0), (1.0, 3.0))  # equal obj0, strictly better obj1
    assert dominates((1.0, 2.0), (2.0, 2.0))


def test_dominates_false_when_worse_in_any_objective():
    assert not dominates((1.0, 3.0), (1.0, 2.0))
    assert not dominates((2.0, 1.0), (1.0, 2.0))  # mixed: neither dominates


def test_dominates_false_when_identical():
    assert not dominates((1.0, 2.0), (1.0, 2.0))


def test_fast_non_dominated_sort_simple_front():
    # A=(1,1) dominates everything; B=(2,2) and C=(3,1) are non-dominated
    # w.r.t. each other (C better on obj1, B better on obj0); D=(3,3) is
    # dominated by both B and C.
    objectives = [(1.0, 1.0), (2.0, 2.0), (3.0, 1.0), (3.0, 3.0)]
    fronts = fast_non_dominated_sort(objectives)

    assert fronts[0] == [0]
    assert set(fronts[1]) == {1, 2}
    assert fronts[2] == [3]


def test_fast_non_dominated_sort_all_non_dominated():
    # Strictly trading off -> everyone in front 0.
    objectives = [(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)]
    fronts = fast_non_dominated_sort(objectives)
    assert len(fronts) == 1
    assert set(fronts[0]) == {0, 1, 2, 3}


def test_crowding_distance_boundary_points_are_infinite():
    front = [(1.0, 5.0), (2.0, 3.0), (3.0, 1.0)]
    distances = crowding_distance(front)
    assert distances[0] == np.inf  # best on objective 0 / worst on objective 1
    assert distances[2] == np.inf  # worst on objective 0 / best on objective 1


def test_crowding_distance_with_exactly_three_points_only_depends_on_range():
    """With only 3 points, the single middle individual's neighbors in
    sorted order are ALWAYS the two boundary (extreme) points, regardless
    of where it actually sits between them -- so its distance depends only
    on the front's total range, not its own position. This is an inherent
    property of the standard crowding-distance formula at n=3, not
    something rul.evolution.nsga.crowding_distance gets to choose."""
    front_off_center = [(1.0, 3.0), (1.1, 2.9), (3.0, 1.0)]  # middle point close to one end
    front_centered = [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]  # middle point equidistant

    d_off_center = crowding_distance(front_off_center)[1]
    d_centered = crowding_distance(front_centered)[1]
    assert d_off_center == d_centered == 2.0


def test_crowding_distance_prefers_sparser_middle_point():
    # A front of 4 points is the smallest where a non-boundary individual's
    # *actual* neighbors (not just the front's extremes) determine its
    # distance, so this can genuinely distinguish "near a neighbor" from
    # "centered between its neighbors".
    front_clustered = [(1.0, 4.0), (2.0, 3.0), (2.1, 2.9), (4.0, 1.0)]  # idx1 packed near idx2
    front_spread = [(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)]  # idx1 evenly spaced

    d_clustered = crowding_distance(front_clustered)[1]
    d_spread = crowding_distance(front_spread)[1]
    assert d_spread > d_clustered


def test_crowding_distance_two_points_both_infinite():
    distances = crowding_distance([(1.0, 2.0), (2.0, 1.0)])
    assert np.all(distances == np.inf)


def test_crowding_distance_handles_inf_objective_without_nan():
    """rul.evolution.fitness.evaluate_genome returns val_rmse=inf for a
    genome that produced zero training windows -- a real, reachable
    fitness value, not just a hypothetical. Mixing it into a front must not
    produce NaN distances (silently corrupting selection) or a
    divide-by-zero warning."""
    front = [(1.0, 100.0), (2.0, 50.0), (float("inf"), 10.0), (5.0, 30.0)]
    with np.errstate(invalid="raise", divide="raise"):
        distances = crowding_distance(front)
    assert not np.any(np.isnan(distances))


def test_select_next_generation_respects_population_size():
    objectives = [(float(i), float(10 - i)) for i in range(10)]  # all non-dominated
    selected = select_next_generation(objectives, population_size=4)
    assert len(selected) == 4
    assert len(set(selected)) == 4  # no duplicates


def test_select_next_generation_prefers_front_0():
    # front 0: indices 0,1; front 1 (dominated): index 2
    objectives = [(1.0, 2.0), (2.0, 1.0), (3.0, 3.0)]
    selected = select_next_generation(objectives, population_size=2)
    assert set(selected) == {0, 1}

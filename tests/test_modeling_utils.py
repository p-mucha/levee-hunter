import numpy as np

from levee_hunter.modeling.modeling_utils import count_pred_pixels_within_distance


def test_count_pred_pixels_within_distance():

    example_mask1 = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype="uint8",
    )

    example_mask2 = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
        ],
        dtype="uint8",
    )
    assert count_pred_pixels_within_distance(example_mask1, example_mask2, 0) == 14
    assert count_pred_pixels_within_distance(example_mask1, example_mask2, 1) == 20
    assert count_pred_pixels_within_distance(example_mask1, example_mask2, 2) == 21
    assert count_pred_pixels_within_distance(example_mask1, example_mask2, 3) == 21

    assert count_pred_pixels_within_distance(example_mask2, example_mask1, 0) == 14
    assert count_pred_pixels_within_distance(example_mask2, example_mask1, 1) == 19
    assert count_pred_pixels_within_distance(example_mask2, example_mask1, 2) == 19
    assert count_pred_pixels_within_distance(example_mask2, example_mask1, 3) == 20
    assert count_pred_pixels_within_distance(example_mask2, example_mask1, 4) == 23
    assert count_pred_pixels_within_distance(example_mask2, example_mask1, 5) == 23

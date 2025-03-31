from affine import Affine
import numpy as np
import rioxarray
import xarray as xr

from levee_hunter.processing.processing_utils import (
    split_images,
    remove_empty_images,
    remove_invalid_images,
)


def create_dummy_tif(add_invalid: bool = False) -> xr.DataArray:
    """
    Create a dummy single-band tif image as an xarray.DataArray
    with shape (band=1, y=10, x=10).
    """
    if add_invalid:
        data = np.zeros((10, 10), dtype=np.float32)
    else:
        data = np.zeros((10, 10), dtype=np.uint8)

    # Draw a horizontal line at row 5 with mixed values:
    data[5, :5] = 3  # First half of the row set to 3
    data[5, 5:] = 5  # Second half of the row set to 5

    # Draw a vertical line at column 3 with mixed values:
    data[:3, 3] = 3  # Top part of the column set to 3
    data[3:, 3] = 5  # Bottom part of the column set to 5

    # Additional points with different values:
    data[4, 8] = 3
    data[3, 8] = 5

    if add_invalid:
        # Add some invalid pixels, in Lidar data these are set to very
        # large negative values, so we will use -9999999
        # to represent invalid pixels.
        data[0, 0] = -9999999
        data[9, 9] = -9999999

    # Reshape to (band, y, x) = (1, 10, 10)
    data_3d = data.reshape(1, 10, 10)

    # Create the DataArray with three dimensions
    da = xr.DataArray(
        data_3d,
        dims=["band", "y", "x"],
        coords={"band": [1], "y": np.arange(10), "x": np.arange(10)},
    )

    # Define an affine transform: upper-left corner at (0,10), pixel size = 1
    transform = Affine(1, 0, 0, 0, 1, 0)
    da.rio.write_transform(transform, inplace=True)
    da.rio.write_crs("EPSG:4326", inplace=True)

    return da


def test_split_images_5_0():
    """
    Test the split_images function, for case where we have 10x10 image and mask, and we split
    into 4 5x5 images with no overlap.
    """
    tif_image = create_dummy_tif()
    mask = np.array(
        [
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
            ]
        ],
        dtype="uint8",
    )

    expected_mask_1 = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
            ]
        ],
        dtype="uint8",
    )

    expected_mask3 = np.array(
        [
            [
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ],
        dtype="uint8",
    )

    expected_smaller_img_1 = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 5, 0],
                [0, 0, 0, 3, 0],
            ]
        ],
        dtype="uint8",
    )

    expected_smaller_img_2 = np.array(
        [
            [
                [3, 3, 3, 5, 3],
                [0, 0, 0, 5, 0],
                [0, 0, 0, 5, 0],
                [0, 0, 0, 5, 0],
                [0, 0, 0, 5, 0],
            ]
        ],
        dtype="uint8",
    )

    # this is case where we split single 10x10 image into 4 images, with no overlap
    # So the input mask and image are (1, 10, 10) and we expect 4 images of shape (1, 5, 5)
    # and 4 masks of shape (1, 5, 5)
    smaller_images, smaller_masks = split_images(
        images=tif_image, masks=mask, final_size=5, overlap=0
    )

    assert len(smaller_images) == 4, "Should have split into 4 images."
    assert len(smaller_masks) == 4, "Should have split into 4 masks."
    assert smaller_images[0].shape == (1, 5, 5), "Each smaller image should be 1x5x5."
    assert smaller_masks[0].shape == (1, 5, 5), "Each smaller mask should be 1x5x5."
    assert np.allclose(
        smaller_images[0].values, tif_image.values[:, 0:5, 0:5]
    ), "First smaller image should match top-left corner of original."
    assert np.allclose(
        smaller_masks[0], mask[:, 0:5, 0:5]
    ), "First smaller mask should match top-left corner of original."
    assert smaller_images[1].shape == (1, 5, 5), "Each smaller image should be 1x5x5."
    assert smaller_masks[1].shape == (1, 5, 5), "Each smaller mask should be 5x5."

    # test with previously verified values
    assert np.allclose(
        smaller_images[1].values, expected_smaller_img_1
    ), "1st smaller image should match expected values"
    assert np.array_equal(
        smaller_masks[1], expected_mask_1
    ), "1st smaller mask should match expected values"
    assert np.allclose(
        smaller_images[2].values, expected_smaller_img_2
    ), "2nd smaller image should match expected values"
    assert np.array_equal(
        smaller_masks[3], expected_mask3
    ), "3rd smaller mask should match expected values"


def test_split_images_4_0():
    """
    Test the split_images function, for case where we have 10x10 image and mask, and we split
    into 4 4x4 images with no overlap. This will obviously not cover the entire original image.
    """
    tif_image = create_dummy_tif()
    mask = np.array(
        [
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
            ]
        ],
        dtype="uint8",
    )

    expected_mask_1 = np.array(
        [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype="int8"
    )

    expected_mask3 = np.array(
        [[[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype="uint8"
    )

    expected_smaller_img_1 = np.array(
        [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype="uint8"
    )

    expected_smaller_img_2 = np.array(
        [[[0, 0, 0, 5], [3, 3, 3, 5], [0, 0, 0, 5], [0, 0, 0, 5]]], dtype="uint8"
    )

    # this is case where we split single 10x10 image into 4 images 4x4, with no overlap
    # So the input mask and image are (1, 10, 10) and we expect 4 images of shape (1, 4, 4)
    # and 4 masks of shape (1, 4, 4)
    smaller_images, smaller_masks = split_images(
        images=tif_image, masks=mask, final_size=4, overlap=0
    )

    assert len(smaller_images) == 4, "Should have split into 4 images."
    assert len(smaller_masks) == 4, "Should have split into 4 masks."
    assert smaller_images[0].shape == (1, 4, 4), "Each smaller image should be 1x5x5."
    assert smaller_masks[0].shape == (1, 4, 4), "Each smaller mask should be 1x5x5."
    assert np.allclose(
        smaller_images[0].values, tif_image.values[:, 0:4, 0:4]
    ), "First smaller image should match top-left corner of original."
    assert np.allclose(
        smaller_masks[0], mask[:, 0:4, 0:4]
    ), "First smaller mask should match top-left corner of original."
    assert smaller_images[1].shape == (1, 4, 4), "Each smaller image should be 1x5x5."
    assert smaller_masks[1].shape == (1, 4, 4), "Each smaller mask should be 5x5."

    # test with previously verified values
    assert np.allclose(
        smaller_images[1].values, expected_smaller_img_1
    ), "1st smaller image should match expected values"
    assert np.array_equal(
        smaller_masks[1], expected_mask_1
    ), "1st smaller mask should match expected values"
    assert np.allclose(
        smaller_images[2].values, expected_smaller_img_2
    ), "2nd smaller image should match expected values"
    assert np.array_equal(
        smaller_masks[3], expected_mask3
    ), "3rd smaller mask should match expected values"


def test_split_images_3_1():
    """
    Test the split_images function, for case where we have 10x10 image and mask, and we split
    into 3x3 images with 1 overlap.
    """
    tif_image = create_dummy_tif()
    mask = np.array(
        [
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
            ]
        ],
        dtype="uint8",
    )

    smaller_images, smaller_masks = split_images(
        images=tif_image, masks=mask, final_size=3, overlap=1
    )

    assert len(smaller_images) == 16, "Should have split into 16 images."
    assert len(smaller_masks) == 16, "Should have split into 16 masks."
    assert smaller_images[0].shape == (1, 3, 3), "Each smaller image should be 1x3x3."
    assert smaller_masks[0].shape == (1, 3, 3), "Each smaller mask should be 1x3x3."

    assert (
        np.max(smaller_images[0].values) == 0
    ), "First smaller image should be all zeros."
    assert np.max(smaller_masks[0]) == 0, "First smaller mask should be all zeros."
    assert np.allclose(
        smaller_images[1].values,
        np.array([[[0, 3, 0], [0, 3, 0], [0, 3, 0]]], dtype="uint8"),
    )

    assert np.allclose(
        smaller_masks[1], np.array([[[0, 1, 0], [0, 1, 0], [0, 1, 0]]], dtype="uint8")
    )

    assert (
        np.max(smaller_images[2].values) == 0
    ), "Third smaller image should be all zeros."
    assert np.max(smaller_masks[2]) == 0, "Third smaller mask should be all zeros."

    assert np.allclose(
        smaller_images[5].values,
        np.array([[[0, 3, 0], [0, 5, 0], [0, 5, 0]]], dtype="uint8"),
    )

    assert np.allclose(
        smaller_masks[5], np.array([[[0, 1, 0], [0, 1, 0], [0, 1, 0]]], dtype="uint8")
    )

    assert (
        np.max(smaller_images[6].values) == 0
    ), "Sixth smaller image should be all zeros."
    assert np.max(smaller_masks[6]) == 0, "Sixth smaller mask should be all zeros."

    assert np.allclose(
        smaller_images[7].values,
        np.array([[[0, 0, 0], [0, 0, 5], [0, 0, 3]]], dtype="uint8"),
    )
    assert np.allclose(
        smaller_masks[7], np.array([[[0, 0, 0], [0, 0, 1], [0, 0, 1]]], dtype="uint8")
    )

    assert np.allclose(
        smaller_images[8].values,
        np.array([[[0, 0, 0], [3, 3, 3], [0, 0, 0]]], dtype="uint8"),
    )
    assert np.allclose(
        smaller_masks[8], np.array([[[0, 0, 0], [1, 1, 1], [0, 0, 0]]], dtype="uint8")
    )

    assert np.allclose(
        smaller_images[9].values,
        np.array([[[0, 5, 0], [3, 5, 3], [0, 5, 0]]], dtype="uint8"),
    )
    assert np.allclose(
        smaller_masks[9], np.array([[[0, 1, 0], [1, 1, 1], [0, 1, 0]]], dtype="uint8")
    )


def test_remove_invalid_images():
    """
    Test the remove_invalid_images function.
    """
    tif_image = create_dummy_tif(add_invalid=True)
    mask = np.array(
        [
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
            ]
        ],
        dtype="uint8",
    )

    smaller_images, smaller_masks = split_images(
        images=tif_image, masks=mask, final_size=3, overlap=1
    )

    smaller_images, smaller_masks = remove_invalid_images(
        images=smaller_images, masks=smaller_masks
    )

    assert (
        len(smaller_images) == 15
    ), "We split into 16 parts, after removing invalid images, 15 should be left"
    assert (
        len(smaller_masks) == 15
    ), "We split into 16 parts, after removing invalid images, 15 should be left"
    assert smaller_images[0].shape == (1, 3, 3), "Each smaller image should be 1x3x3."
    assert smaller_masks[0].shape == (1, 3, 3), "Each smaller mask should be 1x3x3."

    assert np.allclose(
        smaller_images[0].values,
        np.array([[[0, 3, 0], [0, 3, 0], [0, 3, 0]]], dtype="uint8"),
    )
    assert np.allclose(
        smaller_masks[0], np.array([[[0, 1, 0], [0, 1, 0], [0, 1, 0]]], dtype="uint8")
    )


def test_remove_empty_images():
    tif_image = create_dummy_tif(add_invalid=False)
    mask = np.array(
        [
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
            ]
        ],
        dtype="uint8",
    )
    # invert
    mask = 1 - mask

    smaller_images, smaller_masks = split_images(
        images=tif_image, masks=mask, final_size=3, overlap=1
    )

    smaller_images, smaller_masks = remove_empty_images(
        images=smaller_images, masks=smaller_masks, keep_empty=0.5, inverted=True
    )

    assert (
        len(smaller_images) == 12
    ), "We split into 16 parts, after removing empty images, 12 should be left"
    assert (
        len(smaller_masks) == 12
    ), "We split into 16 parts, after removing empty images, 12 should be left"

    smaller_images, smaller_masks = remove_empty_images(
        images=smaller_images, masks=smaller_masks, keep_empty=0.5, inverted=True
    )
    assert (
        len(smaller_images) == 12
    ), "Using remove_empty_images again should not change the number of images"
    assert (
        len(smaller_masks) == 12
    ), "Using remove_empty_images again should not change the number of masks"

    smaller_images, smaller_masks = remove_empty_images(
        images=smaller_images, masks=smaller_masks, keep_empty=0.3, inverted=True
    )
    assert (
        len(smaller_images) == 10
    ), "We split into 16 parts, after removing empty images with keep_empty=0.3, 10 should be left"
    assert (
        len(smaller_masks) == 10
    ), "We split into 16 parts, after removing empty images with kee_empty=0.3, 10 should be left"

    smaller_images, smaller_masks = remove_empty_images(
        images=smaller_images, masks=smaller_masks, keep_empty=0.2, inverted=True
    )
    assert (
        len(smaller_images) == 9
    ), "We split into 16 parts, after removing empty images with keep_empty=0.2, 9 should be left"
    assert (
        len(smaller_masks) == 9
    ), "We split into 16 parts, after removing empty images with kee_empty=0.2, 9 should be left"

    smaller_images, smaller_masks = remove_empty_images(
        images=smaller_images, masks=smaller_masks, keep_empty=0.0, inverted=True
    )
    assert (
        len(smaller_images) == 8
    ), "We split into 16 parts, after removing empty images with keep_empty=0.0, 8 should be left"
    assert (
        len(smaller_masks) == 8
    ), "We split into 16 parts, after removing empty images with kee_empty=0.0, 8 should be left"

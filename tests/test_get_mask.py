from affine import Affine
import geopandas as gpd
import numpy as np
import rioxarray
from shapely.geometry import LineString, box, MultiLineString
import xarray as xr

from levee_hunter.get_mask import get_mask


def create_dummy_tif():
    """
    Create a dummy single-band tif image as an xarray.DataArray
    with shape (band=1, y=10, x=10).
    """
    data = np.zeros((10, 10), dtype=np.uint8)

    data[5, :5] = 3
    data[5, 5:] = 5

    data[:3, 3] = 3
    data[3:, 3] = 5

    data[4, 8] = 3
    data[3, 8] = 5

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


def create_dummy_levees_linestring():
    """
    Create a dummy GeoDataFrame with levee geometries.
    Each LineString contains a point for each coordinate along the line.
    This function creates:
      - A horizontal line along row 5 (x from 0 to 9),
      - A vertical line along column 3 (y from 0 to 9),
      - An extra short segment covering the points (8,3) and (8,4).
    """
    # Horizontal line: all points along row 5, x=0 to 9
    horizontal_coords = [(x, 5) for x in range(10)]
    horizontal_line = LineString(horizontal_coords)

    # Vertical line: all points along column 3, y=0 to 9
    vertical_coords = [(3, y) for y in range(10)]
    vertical_line = LineString(vertical_coords)

    # Extra segment: explicitly include both points (8,3) and (8,4)
    extra_line = LineString([(8, 3), (8, 4)])

    # Combine into a MultiLineString
    multi_line = MultiLineString([horizontal_line, vertical_line, extra_line])

    levees = gpd.GeoDataFrame({"geometry": [multi_line]}, crs="EPSG:4326")
    return levees


def test_get_mask_dilated_linestring():
    """
    Test the get_mask() function using a Linestring levee.
    We use the "dilated" mask type. We do not test the gaussian option here.
    """
    tif_image = create_dummy_tif()
    levees = create_dummy_levees_linestring()

    # Call get_mask() with mask_type "dilated". Use a known dilation_size (e.g., 3) for reproducibility.
    image_out, mask_out = get_mask(
        tif_image,
        levees,
        invert=False,
        mask_type="dilated",
        dilation_size=1,
        gaussian_sigma=5.0,
    )

    # Here use the inverted option.
    image_out_inverted, mask_out_inverted = get_mask(
        tif_image,
        levees,
        invert=True,
        mask_type="dilated",
        dilation_size=1,
        gaussian_sigma=5.0,
    )

    # Assert that the output image is the same as the tif image's values.
    np.testing.assert_array_equal(image_out, tif_image.values)
    mask_out_expected = np.array(
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
    )

    assert np.allclose(
        image_out,
        image_out_inverted,
        atol=1e-6,
    ), "get_mask should return the same image for both inverted and non-inverted cases."
    assert np.allclose(
        image_out,
        tif_image.values,
        atol=1e-6,
    ), "Image should be the same as the original tif image values."
    assert mask_out.dtype == np.uint8, "Mask should have dtype uint8."
    assert np.all(np.isin(mask_out, [0, 1])), "Mask must be binary (only 0 and 1)."
    assert np.array_equal(
        mask_out, mask_out_expected
    ), "Mask should contain some ones where the levee is rasterized."
    assert np.array_equal(
        mask_out_inverted, 1 - mask_out
    ), "Inverted mask should be the complement of the original mask."
    assert mask_out_inverted.dtype == np.uint8, "Inverted mask should have dtype uint8."

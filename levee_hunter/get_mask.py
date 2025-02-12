import xarray
from typing import Tuple
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation
import numpy as np
from shapely.geometry import box
import geopandas as gpd


def get_mask(
    tif_image: xarray.DataArray,
    levees: gpd.GeoDataFrame,
    invert: bool = False,
    dilation_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a tif image and a geopandas dataframe of levees, return the image and the mask.

    Parameters:
    - tif_image: A rioxarray DataArray containing the raster data.
    - levees: A GeoDataFrame containing levee geometries.
    - invert: If True, inverts the mask.
    - dilation_size: Size of the kernel used for thickening levee lines (default=10).

    Returns:
    - A tuple containing:
      1. The original raster image as a NumPy array.
      2. The binary mask as a NumPy array.
    """
    transform = tif_image.rio.transform()  # Affine transformation
    shape = tif_image.shape[-2:]  # Shape of the raster (rows, cols)

    # Filter levees that intersect the tif extent
    extent_geom = tif_image.rio.bounds()  # Bounds of the tif image
    extent_box = box(*extent_geom)  # Define bounding box as shapely geometry
    levees_in_extent = levees[levees.intersects(extent_box)]

    # Rasterize levees onto the same grid as the tif
    levee_raster = rasterize(
        [(geom, 1) for geom in levees_in_extent.geometry],  # Geometry and value
        out_shape=shape,
        transform=transform,
        fill=0,  # Default value for areas without levees
        dtype="uint8",
    )

    # Thicken the white lines using binary dilation
    structure = np.ones((dilation_size, dilation_size), dtype=bool)
    thickened_raster = binary_dilation(levee_raster, structure=structure).astype(
        np.uint8
    )

    if invert:
        thickened_raster = 1 - thickened_raster

    return tif_image.values, thickened_raster

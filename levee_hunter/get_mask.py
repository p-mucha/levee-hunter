import xarray
from typing import Tuple
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, gaussian_filter
import numpy as np
from shapely.geometry import box
import geopandas as gpd


def get_mask(
    tif_image: xarray.DataArray,
    levees: gpd.GeoDataFrame,
    invert: bool = False,
    mask_type: str = "dilated",
    dilation_size: int = 10,
    gaussian_sigma: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a tif image and a geopandas dataframe of levees, return the image and the mask.

    Parameters:
    - tif_image: A rioxarray DataArray containing the raster data.
    - levees: A GeoDataFrame containing levee geometries.
    - invert: If True, inverts the mask.
    - mask_type: Type of mask to apply ("dilated" or "gaussian")
    - dilation_size: Size of the kernel used for thickening levee lines (default=10).
    - gaussian_sigma: Standard deviation for Gaussian filter (default=5.0)

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

    if mask_type == "dilated":
        # Apply binary dilation
        structure = np.ones((dilation_size, dilation_size), dtype=bool)
        mask = binary_dilation(levee_raster, structure=structure).astype(np.uint8)
    elif mask_type == "gaussian":
        # Apply Gaussian filter
        mask = gaussian_filter(levee_raster.astype(float), sigma=gaussian_sigma)
        mask = (mask > 0.1).astype(np.uint8)  # Convert to binary mask
    else:
        raise ValueError("Invalid mask_type. Choose 'dilated' or 'gaussian'.")

    if invert:
        mask = 1 - mask

    return tif_image.values, mask

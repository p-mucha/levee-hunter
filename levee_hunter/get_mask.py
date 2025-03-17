import xarray
from typing import Tuple
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, gaussian_filter
import numpy as np
from shapely.geometry import box
import geopandas as gpd
import warnings


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
    - mask_type: Type of mask to apply ("dilated" or "gaussian", or None for no change)
    - dilation_size: Size of the kernel used for thickening levee lines (default=10).
    - gaussian_sigma: Standard deviation for Gaussian filter (default=5.0)

    Returns:
    - A tuple containing:
      1. The original raster image as a NumPy array.
      2. The binary mask as a NumPy array.
    """
    valid_mask_types = {"dilated", "gaussian"}
    if mask_type not in valid_mask_types and mask_type is not None:
        raise ValueError(
            f"Invalid mask_type. Choose one of {valid_mask_types}, or None."
        )

    if dilation_size != 10 and mask_type != "dilated":
        warnings.warn("dilation_size will be ignored if mask_type is not 'dilated'.")

    if gaussian_sigma != 5.0 and mask_type != "gaussian":
        warnings.warn("gaussian_sigma will be ignored if mask_type is not 'gaussian'.")

    if mask_type is None:
        if dilation_size != 10:
            warnings.warn("dilation_size will be ignored if mask_type is None.")
        if gaussian_sigma != 5.0:
            warnings.warn("gaussian_sigma will be ignored if mask_type is None.")

    if not isinstance(tif_image, xarray.DataArray):
        raise ValueError("tif_image must be an xarray.DataArray.")

    # Check if the CRS of the levees is the same as the tif image
    # When working on version 2.0, suddenly reprojecting levees
    # was extremely slow. I therefore decided to reproject the tif
    # instead. This is used to find levees in extent,
    # next we reproject those levees which should be fast as those should be
    # a few levees usually. # <- those lines are affected by this temp fix
    if levees.crs != tif_image.rio.crs:
        tif_image_reprojected = tif_image.rio.reproject(levees.crs)  # <-

    transform = tif_image.rio.transform()  # Affine transformation
    shape = tif_image.shape[-2:]  # Shape of the raster (rows, cols)

    # Filter levees that intersect the tif extent
    extent_geom = tif_image_reprojected.rio.bounds()  # Bounds of the tif image # <-
    extent_box = box(*extent_geom)  # Define bounding box as shapely geometry
    levees_in_extent = levees[levees.intersects(extent_box)].to_crs(
        tif_image.rio.crs
    )  # <-

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

    # If mask_type is None, return the original raster
    else:
        mask = levee_raster

    if invert:
        mask = 1 - mask

    return tif_image.values, mask

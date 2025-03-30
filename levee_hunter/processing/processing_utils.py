import os
import numpy as np
from typing import Tuple, List
import warnings
import xarray
import rioxarray


def split_images(
    images: list, masks: np.ndarray, final_size: int, overlap: int
) -> Tuple[List[xarray.DataArray], List[np.ndarray]]:
    """Splits the image and mask into smaller patches.

    Inputs:
    - images: Should be either a list of xarray.DataArray, or a single xarray.DataArray.
              Shape of each xarray.DataArray should be (1, H, W).
    - masks: Should be a numpy array, either of shape (N, H, W) or (H, W).
    - final_size: The final size of smaller images after splitting.
    - overlap: The overlap between two patches.

    Outputs:
    - smaller_images: A list of xarray.DataArray, each of shape (1, final_size, final_size).
    - smaller_masks: A list of numpy arrays, each of shape (1, final_size, final_size).
    """

    # If single image provided, convert to list
    if isinstance(images, xarray.DataArray):
        images = [images]

    if isinstance(images, np.ndarray):
        warnings.warn(
            "In the current version, the images should be a list of xarray.DataArray,"
            " but a numpy array was provided. This will only work if images.shape = (N, 1, H, W)."
        )

    # Handle masks now. We decided to keep masks as numpy arrays
    if not isinstance(masks, np.ndarray):
        raise ValueError("Masks should be a numpy array.")

    # masks can potentially have shape (H, W), (N, H, W), or (N, C, H, W)
    # Currently we will assume C = 1
    if len(masks.shape) == 2:
        masks = masks.reshape(1, *masks.shape)

    if len(masks.shape) == 4:
        if masks.shape[1] != 1:
            raise ValueError("The channel dimension of the mask should be 1.")
        masks = masks.squeeze(1)

    smaller_images = []
    smaller_masks = []
    stride = final_size - overlap

    # Loop over images in the list
    for image_no in range(len(images)):
        image = images[image_no]
        mask = masks[image_no]

        # Loop to cover entire image
        for i in range(0, max(image.shape[1] - final_size, 0) + 1, stride):
            for j in range(0, max(image.shape[2] - final_size, 0) + 1, stride):

                # if masks was provided as (N, 1, H, W), the mask would be
                # (1, H, W) since mask = masks[image_no]
                if len(mask.shape) > 2:
                    mask = mask.squeeze()

                # For given image:
                # Divide the image into (final_size, final_size) patches
                # Notice since image = images[image_no], it is a single
                # xarray.DataArray with shape (1, H, W)
                # masks was originally (N, H, W), so mask is (H, W)
                smaller_image = image[:, i : i + final_size, j : j + final_size]
                smaller_mask = mask[i : i + final_size, j : j + final_size]

                # Reshape to (1, H, W), no need to do it for smaller_image,
                # since it is already in that shape
                smaller_mask = smaller_mask.reshape(1, *smaller_mask.shape)

                smaller_images.append(smaller_image)
                smaller_masks.append(smaller_mask)

    return smaller_images, smaller_masks


def remove_invalid_images(
    images: List[xarray.DataArray], masks: List[np.ndarray]
) -> Tuple[List[xarray.DataArray], List[np.ndarray]]:
    """
    Removes images where the minimum pixel value is less than -9999,
    along with their corresponding masks.

    Inputs:
    - images: A list of xarray.DataArray, each being (1, H, W).
    - masks: A list of numpy arrays, each being (1, H, W).

    Outputs:
    - images: A list of xarray.DataArray, each being (1, H, W).
    - masks: A list of numpy arrays, each being (1, H, W).

    This might happen if data is missing in a given region.
    Basically this function allows (after splitting large image into smaller ones),
    to keep some parts of the original image even if there are parts missing on it,
    while removing any smaller images that might contain those missing parts.
    """

    # All potential error checks
    if not isinstance(images, list):
        raise ValueError("Images should be a list of xarray.DataArray.")
    if not isinstance(masks, list):
        raise ValueError("Masks should be a list of numpy arrays.")
    if not len(images) == len(masks):
        raise ValueError("Images and masks should have the same length.")
    if len(images) != 0 and not isinstance(images[0], xarray.DataArray):
        raise ValueError("Each element of images should be a xarray.DataArray.")
    if len(masks) != 0 and not isinstance(masks[0], np.ndarray):
        raise ValueError("Each element of masks should be a numpy array.")

    # Images is a list of xarray.DataArray, each being (1, H, W)
    # Convert it to a numpy array of shape (N, 1, H, W) for check
    images_arr = np.array([image.values for image in images])

    # Find indices where the min value in the image is >= -9999 (valid images)
    valid_indices = np.where(np.min(images_arr, axis=(1, 2, 3)) >= -9999)[0]

    # Keep only valid images and their corresponding masks
    images = [images[i] for i in valid_indices]
    masks = [masks[i] for i in valid_indices]

    return images, masks


def remove_empty_images(
    images: List[xarray.DataArray],
    masks: List[np.ndarray],
    keep_empty: float = 0.2,
    inverted: bool = True,
) -> Tuple[List[xarray.DataArray], List[np.ndarray]]:
    """
    Removes images where the mask is empty (no levee present),
    along with their corresponding masks. If keep_empty > 0.0,
    a fraction of empty images will be added back to the
    non-empty images.

    Inputs:
    - images: A list of xarray.DataArray, each being (1, H, W).
    - masks: A list of numpy arrays, each being (1, H, W).

    Outputs:
    - images: A list of xarray.DataArray, each being (1, H, W).
    - masks: A list of numpy arrays, each being (1, H, W).
    """

    # All potential error checks
    if not isinstance(images, list):
        raise ValueError("Images should be a list of xarray.DataArray.")
    if not isinstance(masks, list):
        raise ValueError("Masks should be a list of numpy arrays.")
    if not len(images) == len(masks):
        raise ValueError("Images and masks should have the same length.")
    if len(images) != 0 and not isinstance(images[0], xarray.DataArray):
        raise ValueError("Each element of images should be a xarray.DataArray.")
    if len(masks) != 0 and not isinstance(masks[0], np.ndarray):
        raise ValueError("Each element of masks should be a numpy array.")

    # masks is a list of N np.ndarrays, each being (1, H, W)
    # Convert it to a numpy array of shape (N, 1, H, W) for check
    masks = np.array(masks)

    # Find indices where the min value in the image is 1
    # Assuming target pixels are 0, this means there is no levee
    # on mask, so this image is empty
    # If inverted==False, the situation is reversed and for empty we would have
    # max pixel being 0 (no 1 which would be target)
    if inverted:
        empty_indices = np.where(np.min(masks, axis=(1, 2, 3)) == 1)[0]
    else:
        empty_indices = np.where(np.max(masks, axis=(1, 2, 3)) == 0)[0]

    # Get non-empty and empty images and masks
    empty_images = [images[i] for i in empty_indices]
    non_empty_images = [images[i] for i in range(len(images)) if i not in empty_indices]
    empty_masks = [masks[i] for i in empty_indices]
    non_empty_masks = [masks[i] for i in range(len(masks)) if i not in empty_indices]

    # Add back a fraction of empty images to the non_empty_images
    # For example if there are 100 non empty, and keep_empty=0.2
    # we would add 20 empty images back
    if keep_empty > 0.0:
        num_to_add = int(len(non_empty_images) * keep_empty)
        if num_to_add > 0:
            empty_images_to_add = empty_images[:num_to_add]
            empty_masks_to_add = empty_masks[:num_to_add]

            non_empty_images += empty_images_to_add
            non_empty_masks += empty_masks_to_add

    assert len(non_empty_images) == len(non_empty_masks)

    return non_empty_images, non_empty_masks


def filter_single_image_by_overlap(
    tif_img: xarray.DataArray, bounds_file: str, threshold: int = 25
) -> bool:
    """
    Filters a single new image based on overlap with existing accepted image bounds.

    Parameters:
    - tif_img (xarray.DataArray): The new image to be checked for overlap.
    - bounds_file (str): Path to the text file storing accepted image bounds.
    - threshold (int, optional): The percentage overlap threshold to filter images. Default is 25.

    Returns:
    - bool: True if the image is accepted, False if rejected.
    """

    # Example usage:
    # tif_img = rioxarray.open_rasterio("/path/to/image.tif")
    # bounds_file = "/path/to/bounds.txt"
    # result = filter_single_image_by_overlap(image_path, bounds_file)

    def calculate_overlap_percentage(bounds1, bounds2):
        overlap_xmin = max(bounds1[0], bounds2[0])
        overlap_ymin = max(bounds1[1], bounds2[1])
        overlap_xmax = min(bounds1[2], bounds2[2])
        overlap_ymax = min(bounds1[3], bounds2[3])

        if overlap_xmin < overlap_xmax and overlap_ymin < overlap_ymax:
            overlap_area = (overlap_xmax - overlap_xmin) * (overlap_ymax - overlap_ymin)
            area1 = (bounds1[2] - bounds1[0]) * (bounds1[3] - bounds1[1])
            area2 = (bounds2[2] - bounds2[0]) * (bounds2[3] - bounds2[1])
            percentage_overlap1 = (overlap_area / area1) * 100
            percentage_overlap2 = (overlap_area / area2) * 100
            return max(percentage_overlap1, percentage_overlap2)
        return 0

    # Check image CRS and get bounds
    if tif_img.rio.crs.to_epsg() != 5070:
        tif_img = tif_img.rio.reproject("EPSG:5070")

    new_bounds = tif_img.rio.bounds()

    # Load existing bounds from the bounds file (if it exists)
    # if file exists, read the bounds and check for overlap
    # if it does not exist, we can accept the image
    existing_bounds_list = []
    if os.path.exists(bounds_file):
        with open(bounds_file, "r") as f:
            for line in f:
                existing_bounds_list.append(tuple(map(float, line.strip().split(","))))

    # If this list is non-empty, we need to check for overlap
    if existing_bounds_list:
        # Compare the new image with existing bounds
        for existing_bounds in existing_bounds_list:
            overlap_percentage = calculate_overlap_percentage(
                new_bounds, existing_bounds
            )
            if overlap_percentage > threshold:
                print(f"Image rejected due to {overlap_percentage:.2f}% overlap.")
                return False

    # If we reached this step, file either does not exist or there is no significant overlap
    print(f"Image passed overlap check.")
    return True

import os
import numpy as np
from typing import Tuple
import warnings
import xarray
import rioxarray


def split_images(
    images: list, targets: np.ndarray, final_size: int, overlap: int
) -> Tuple[list, list]:
    """Splits the image and mask into smaller patches.

    Inputs:
    - images: Should be either a list of xarray.DataArray, or a single xarray.DataArray.
              Shape of each xarray.DataArray should be (1, H, W).
    - targets: Should be a numpy array, either of shape (N, H, W) or (H, W).
    - final_size: The final size of smaller images after splitting.
    - overlap: The overlap between two patches.

    Outputs:
    - smaller_images: A list of xarray.DataArray, each of shape (1, final_size, final_size).
    - smaller_targets: A list of numpy arrays, each of shape (1, final_size, final_size).
    """

    # If single image provided, convert to list
    if isinstance(images, xarray.DataArray):
        images = [images]

    if isinstance(images, np.ndarray):
        warnings.warn(
            "In the current version, the images should be a list of xarray.DataArray,"
            " but a numpy array was provided. This will only work if images.shape = (N, 1, H, W)."
        )

    # Handle targets now. We decided to keep targets as numpy arrays
    if not isinstance(targets, np.ndarray):
        raise ValueError("Targets should be a numpy array.")

    # targets can potentially have shape (H, W), (N, H, W), or (N, C, H, W)
    # Currently we will assume C = 1
    if len(targets.shape) == 2:
        targets = targets.reshape(1, *targets.shape)

    if len(targets.shape) == 4:
        if targets.shape[1] != 1:
            raise ValueError("The channel dimension of the target should be 1.")
        targets = targets.squeeze(1)

    smaller_images = []
    smaller_targets = []
    stride = final_size - overlap

    # Loop over images in the list
    for image_no in range(len(images)):
        image = images[image_no]
        target = targets[image_no]

        # Loop to cover entire image
        for i in range(0, max(image.shape[1] - final_size, 0) + 1, stride):
            for j in range(0, max(image.shape[2] - final_size, 0) + 1, stride):

                # For given image:
                # Divide the image into (final_size, final_size) patches
                # Notice since image = images[image_no], it is a single
                # xarray.DataArray with shape (1, H, W)
                # targets was originally (N, H, W), so target is (H, W)
                smaller_image = image[:, i : i + final_size, j : j + final_size]
                smaller_target = target[i : i + final_size, j : j + final_size]

                # Reshape to (1, H, W), no need to do it for smaller_image,
                # since it is already in that shape
                smaller_target = smaller_target.reshape(1, *smaller_target.shape)

                smaller_images.append(smaller_image)
                smaller_targets.append(smaller_target)

    return smaller_images, smaller_targets


def remove_invalid_images(images: list, targets: list) -> Tuple[list, list]:
    """
    Removes images where the minimum pixel value is less than -9999,
    along with their corresponding targets.

    Inputs:
    - images: A list of xarray.DataArray, each being (1, H, W).
    - targets: A list of numpy arrays, each being (1, H, W).

    Outputs:
    - images: A list of xarray.DataArray, each being (1, H, W).
    - targets: A list of numpy arrays, each being (1, H, W).

    This might happen if data is missing in a given region.
    Basically this function allows (after splitting large image into smaller ones),
    to keep some parts of the original image even if there are parts missing on it,
    while removing any smaller images that might contain those missing parts.
    """

    # All potential error checks
    if not isinstance(images, list):
        raise ValueError("Images should be a list of xarray.DataArray.")
    if not isinstance(targets, list):
        raise ValueError("Targets should be a list of numpy arrays.")
    if not len(images) == len(targets):
        raise ValueError("Images and targets should have the same length.")
    if len(images) != 0 and not isinstance(images[0], xarray.DataArray):
        raise ValueError("Each element of images should be a xarray.DataArray.")
    if len(targets) != 0 and not isinstance(targets[0], np.ndarray):
        raise ValueError("Each element of targets should be a numpy array.")

    # Images is a list of xarray.DataArray, each being (1, H, W)
    # Convert it to a numpy array of shape (N, 1, H, W) for check
    images_arr = np.array([image.values for image in images])

    # Find indices where the min value in the image is >= -9999 (valid images)
    valid_indices = np.where(np.min(images_arr, axis=(1, 2, 3)) >= -9999)[0]

    # Keep only valid images and their corresponding targets
    images = [images[i] for i in valid_indices]
    targets = [targets[i] for i in valid_indices]

    return images, targets


def remove_empty_images(
    images: list, targets: list, keep_empty: float = 0.2, inverted: bool = True
) -> Tuple[list, list]:
    """
    Removes images where the target is empty (no levee present),
    along with their corresponding targets. If keep_empty > 0.0,
    a fraction of empty images will be added back to the
    non-empty images.

    Inputs:
    - images: A list of xarray.DataArray, each being (1, H, W).
    - targets: A list of numpy arrays, each being (1, H, W).

    Outputs:
    - images: A list of xarray.DataArray, each being (1, H, W).
    - targets: A list of numpy arrays, each being (1, H, W).
    """

    # All potential error checks
    if not isinstance(images, list):
        raise ValueError("Images should be a list of xarray.DataArray.")
    if not isinstance(targets, list):
        raise ValueError("Targets should be a list of numpy arrays.")
    if not len(images) == len(targets):
        raise ValueError("Images and targets should have the same length.")
    if len(images) != 0 and not isinstance(images[0], xarray.DataArray):
        raise ValueError("Each element of images should be a xarray.DataArray.")
    if len(targets) != 0 and not isinstance(targets[0], np.ndarray):
        raise ValueError("Each element of targets should be a numpy array.")

    # targets is a list of N np.ndarrays, each being (1, H, W)
    # Convert it to a numpy array of shape (N, 1, H, W) for check
    targets = np.array(targets)

    # Find indices where the min value in the image is 1
    # Assuming target pixels are 0, this means there is no levee
    # on mask, so this image is empty
    # If inverted==False, the situation is reversed and for empty we would have
    # max pixel being 0 (no 1 which would be target)
    if inverted:
        empty_indices = np.where(np.min(targets, axis=(1, 2, 3)) == 1)[0]
    else:
        empty_indices = np.where(np.max(targets, axis=(1, 2, 3)) == 0)[0]

    # Get non-empty and empty images and targets
    empty_images = [images[i] for i in empty_indices]
    non_empty_images = [images[i] for i in range(len(images)) if i not in empty_indices]
    empty_targets = [targets[i] for i in empty_indices]
    non_empty_targets = [
        targets[i] for i in range(len(targets)) if i not in empty_indices
    ]

    # Add back a fraction of empty images to the non_empty_images
    # For example if there are 100 non empty, and keep_empty=0.2
    # we would add 20 empty images back
    if keep_empty > 0.0:
        num_to_add = int(len(non_empty_images) * keep_empty)
        if num_to_add > 0:
            empty_images_to_add = empty_images[:num_to_add]
            empty_targets_to_add = empty_targets[:num_to_add]

            non_empty_images += empty_images_to_add
            non_empty_targets += empty_targets_to_add

    assert len(non_empty_images) == len(non_empty_targets)

    return non_empty_images, non_empty_targets

def filter_images_by_overlap(new_images_dir, existing_images_dir, threshold=25):
   """
   Filters new images based on overlap with existing images in the filtered_images_dir.
   
   Parameters:
   new_images_dir (str): Directory containing new images to be filtered.
   existing_images_dir (str): Directory containing existing images to compare with.
   threshold (int, optional): The percentage overlap threshold to filter images. Default is 25.
   
   Returns:
   list: A list of filtered new images that do not overlap with existing images beyond the threshold.
   """
   
   # Example usage:
   # new_images_dir = "/path/to/new/images"
   # filtered_images_dir = "/path/to/filtered/images"
   # filtered_new_images = filter_images_by_overlap(new_images_dir, filtered_images_dir, threshold=25)

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

   # Importing the new Lidar data (.tif files)
   new_tif_files = [file for file in os.listdir(new_images_dir) if file.endswith(".tif")]
   # Check all new images are in the CRS EPSG:3717
   for file in new_tif_files:
      img = rioxarray.open_rasterio(os.path.join(new_images_dir, file))
      if img.rio.crs.to_epsg() != 3717:
         print(f"Image {file} is not in the correct CRS (EPSG:3717)")
         return
   # Get the bounds of the new images
   new_tif_bounds = [img.rio.bounds() for img in new_tif_files]

   # Load existing images and get their bounds
   existing_images = [file for file in os.listdir(existing_images_dir) if file.endswith(".tif")]
   existing_bounds_list = [img.rio.bounds() for img in existing_images]

   # Filter new images based on overlap with existing images
   filtered_new_images = []
   filtered_new_bounds_list = []

   for i, new_bounds in enumerate(new_tif_bounds):
      overlap_found = False
      for existing_bounds in existing_bounds_list:
         overlap_percentage = calculate_overlap_percentage(new_bounds, existing_bounds)
         if overlap_percentage > threshold:
            overlap_found = True
            break
      if not overlap_found:
         filtered_new_images.append(new_tif_files[i])
         filtered_new_bounds_list.append(new_bounds)

   print("Number of new images removed:", (len(new_tif_files)-len(filtered_new_images)))

   return filtered_new_images
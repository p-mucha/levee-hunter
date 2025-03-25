import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from typing import Callable
import warnings
import xarray

from levee_hunter.plots import plot_overlayed_img_mask_pred
from levee_hunter.modeling.inference import infer
from levee_hunter.modeling.metrics import custom_metrics
from levee_hunter.augmentations import TRANSFORMS


def model_helper(image: xarray.DataArray, mask: np.ndarray, model: nn.Module) -> None:
    image = image.values.squeeze()
    mask = mask.squeeze()
    augmentation = TRANSFORMS["normalize_only"]
    augmented = augmentation(image=image, mask=mask)

    # those have (H, W) shape now
    aug_image = augmented["image"]
    aug_mask = augmented["mask"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output = infer(
        model, image_tensor=aug_image, device=device, apply_sigmoid=True, threshold=0.5
    )

    plot_overlayed_img_mask_pred(
        image=aug_image,
        mask=aug_mask,
        pred=output,
        figsize=(15, 9),
        cmap="viridis",
        invert=True,
    )

    return None


def model_helper_custom_metrics(
    image: xarray.DataArray, mask: np.ndarray, model: nn.Module
) -> None:
    image = image.values.squeeze()
    mask = mask.squeeze()
    augmentation = TRANSFORMS["normalize_only"]
    augmented = augmentation(image=image, mask=mask)

    # those have (H, W) shape now
    aug_image = augmented["image"]
    aug_mask = augmented["mask"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output = infer(
        model, image_tensor=aug_image, device=device, apply_sigmoid=True, threshold=0.5
    )

    plot_overlayed_img_mask_pred(
        image=aug_image,
        mask=aug_mask,
        pred=output,
        figsize=(15, 9),
        cmap="viridis",
        invert=True,
    )
    custom_metrics(
        targets=1 - aug_mask.reshape(1, *aug_mask.shape),
        predictions=1 - output,
        d=13,
        d_merge=16,
        print_tp_fp_fn=True,
    )

    return None


def bad_overlap_helper(
    image: xarray.DataArray, mask: np.ndarray, bad_bounds_file: str
) -> None:

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
            return percentage_overlap1, percentage_overlap2
        return 0, 0

    # Check image CRS and get bounds
    if image.rio.crs.to_epsg() != 5070:
        image = image.rio.reproject("EPSG:5070")

    new_bounds = image.rio.bounds()

    existing_bounds_list = []
    with open(bad_bounds_file, "r") as f:
        for line in f:
            existing_bounds_list.append(tuple(map(float, line.strip().split(","))))

    # If this list is non-empty, we need to check for overlap
    if existing_bounds_list:

        overlaps1 = []
        overlaps2 = []
        # Compare the new image with existing bounds
        for existing_bounds in existing_bounds_list:
            overlap_percentage1, overlap_percentage2 = calculate_overlap_percentage(
                new_bounds, existing_bounds
            )
            overlaps1.append(overlap_percentage1)
            overlaps2.append(overlap_percentage2)

        overlaps1 = np.array(overlaps1)
        overlaps2 = np.array(overlaps2)
        if np.sum(overlaps1) > 25 and np.max(overlaps2) > 70:
            warnings.warn(
                f"Current image contains area classified as bad.\n Potential total area affected: {np.sum(overlaps1):.2f}%, max overlap on 1m resolution image: {np.max(overlaps2):.2f}%"
            )
    return None


# -------------------------------------- End of Helpers Definitions ------------------------------------------------

HELPERS = {
    "model_helper_custom_metrics": model_helper_custom_metrics,
    "model_helper": model_helper,
    "bad_overlap_helper": bad_overlap_helper,
}


def specify_helper(
    helper_name: str, third_arg
) -> Callable[[xarray.DataArray, np.ndarray], None]:
    """
    Returns a helper function based on the helper_name. Allows to specify the third input, which can be a model or a bounds_file_path.

    Inputs:
    - helper_name (str): The name of the helper function to specify.
    - third_arg: The third argument to pass to the helper function.

    Outputs:
    - a callable on image: xarray.DataArray, mask: np.ndarray, both of which are expected to be (1, H, W) shaped.
    """

    helper_function = HELPERS[helper_name]
    return lambda image, mask: helper_function(image, mask, third_arg)

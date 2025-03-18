import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
import warnings


def apply_mask_type(
    mask: np.ndarray,
    mask_type: str = "dilated",
    dilation_size: int = 10,
    gaussian_sigma: int = 5,
    inverted: bool = True,
) -> np.ndarray:
    """
    Apply a mask type to the original mask with single pixel width levees.

    Inputs:
    - mask: (1, H, W) numpy array of the original mask.
    - mask_type: str, type of mask to apply. Choose from 'dilated' or 'gaussian'.
    - dilation_size: int, size of the dilation kernel.
    - gaussian_sigma: int, standard deviation of the gaussian kernel.
    - inverted: bool, whether the mask is inverted or not, if target pixels are 0, set to True.

    Outputs:
    - mask: (1, H, W) numpy array of the mask with the applied mask type.

    """
    # This block handles potentially wrong option selection by user
    # Unlike get_mask, there is no need to accept None as input here
    valid_mask_types = {"dilated", "gaussian"}
    if mask_type not in valid_mask_types:
        raise ValueError(
            f"Invalid mask_type. Choose one of {valid_mask_types}, or None."
        )
    if dilation_size != 10 and mask_type != "dilated":
        warnings.warn("dilation_size will be ignored if mask_type is not 'dilated'.")
    if gaussian_sigma != 5.0 and mask_type != "gaussian":
        warnings.warn("gaussian_sigma will be ignored if mask_type is not 'gaussian'.")

    if len(mask.shape) > 3:
        warnings.warn("This function is intended for (1, H, W) masks.")

    # Squeeze the mask if it has a channel dimension
    if len(mask.shape) > 2:
        mask = mask.squeeze()

    # Temporarily ivert to apply changes
    if inverted:
        mask = 1 - mask

    if mask_type == "dilated":
        # Apply binary dilation
        structure = np.ones((dilation_size, dilation_size), dtype=bool)
        mask = binary_dilation(mask, structure=structure).astype(np.uint8)

    elif mask_type == "gaussian":
        # Apply Gaussian filter
        mask = gaussian_filter(mask.astype(float), sigma=gaussian_sigma)
        mask = (mask > 0.1).astype(np.uint8)

    # Invert the mask back if it was inverted
    if inverted:
        mask = 1 - mask

    # Add channel dimension back
    return mask.reshape(1, *mask.shape)

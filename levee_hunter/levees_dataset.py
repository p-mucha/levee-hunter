from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import rioxarray
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import overload, Tuple
import warnings
import xarray as xr

from levee_hunter.augmentations import TRANSFORMS
from levee_hunter.processing.apply_mask_type import apply_mask_type


class LeveesDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: str = None,
        weighted: bool = True,
    ) -> None:
        """
        Inputs:
        - images_dir: str, path to the directory containing the images, eg. "../data/processed/1m_1024/images"
        - masks_dir: str, path to the directory containing the masks, eg. "../data/processed/1m_1024/masks"
        - transform: str, name of the transformation from levee_hunter.augmentations.TRANSFORMS to apply to the images
        - weighted: bool, whether to use weighted training or not
        """

        # Convert to Path objects and check if they exist
        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)
        assert images_dir.exists(), f"Path {images_dir} does not exist"
        assert masks_dir.exists(), f"Path {masks_dir} does not exist"

        # convenient to keep those paths
        # useful to create new dataset based on this one for example
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        # Get all image and mask paths
        # Those are paths, not just filenames, like eg:
        # ../data/processed/1m_1024/images/1_6a5a29c4dc_w1.tif
        # We sort to make sure that image and mask at the same index correspond to each other
        self.img_paths = sorted(images_dir.glob("*.tif"))
        self.mask_paths = sorted(masks_dir.glob("*.npy"))

        # Check if there are the same number of images and masks
        assert len(self.img_paths) == len(
            self.mask_paths
        ), "Number of images and masks must be equal"

        self.transform = transform
        # This checks if transform is inside TRANSFORMS dictionary keyes
        # If not, it will raise a warning
        if self.transform and self.transform not in TRANSFORMS:
            warnings.warn(
                "transform should be a string from levee_hunter.augmentations.TRANSFORMS, "
                "code should work without this, but this will be changed in the future",
                FutureWarning,
            )

        self.weighted = weighted

        # weights are kept as lists
        if self.weighted:
            self.weights = self._compute_weights()

    def apply_padding(
        self,
        num_px: int,
    ) -> None:
        """
        Check to apply padding to the images and masks. Images need to be divisible by 32

        Inputs:
        - num_px: int, number of pixels to pad on one side
        """

        # Warn the user about data overwriting
        user_confirmation = input(
            f"WARNING: This operation will overwrite the original mask files in {self.masks_dir}. "
            f"Ensure you have saved a backup if needed. "
            f"Do you want to proceed? (yes/no): "
        )

        if user_confirmation.lower() != "yes":
            print("Operation aborted by user.")
            return

        # Loop over all image paths, ensure dimensions are divisible by 32
        for file_path in tqdm(self.img_paths, desc="Padding images"):
            current_img = rioxarray.open_rasterio(str(file_path))

            # Calculate padding needed for height and width to be divisible by 32
            image_height = current_img.rio.height
            image_width = current_img.rio.width

            pad_height = (32 - image_height % 32) % 32
            pad_width = (32 - image_width % 32) % 32

            if pad_height > 0 or pad_width > 0:
                # Apply padding to make dimensions divisible by 32
                current_img = current_img.pad(
                    {"y": (pad_height, 0), "x": (0, pad_width)},
                    mode="constant",
                    constant_values=-999,
                )
            current_img.rio.to_raster(file_path)

        # Loop over all mask paths, ensure dimensions are divisible by 32
        for file_path in tqdm(self.mask_paths, desc="Padding masks"):
            current_mask = np.load(file_path)

            # Get current dimensions
            if len(current_mask.shape) == 3:  # Shape is (C, H, W)
                mask_height, mask_width = current_mask.shape[1:]
                pad_height = (32 - mask_height % 32) % 32
                pad_width = (32 - mask_width % 32) % 32

                if pad_height > 0 or pad_width > 0:
                    current_mask = np.pad(
                        current_mask,
                        ((0, 0), (pad_height, 0), (0, pad_width)),
                        mode="constant",
                        constant_values=1,  # False
                    )
            else:  # Shape is (H, W)
                mask_height, mask_width = current_mask.shape
                pad_height = (32 - mask_height % 32) % 32
                pad_width = (32 - mask_width % 32) % 32

                if pad_height > 0 or pad_width > 0:
                    current_mask = np.pad(
                        current_mask,
                        ((pad_height, 0), (0, pad_width)),
                        mode="constant",
                        constant_values=1,  # False
                    )

            # Save the modified mask back to the original path
            np.save(file_path, current_mask)

    def remove_padding(self, original_px: int) -> None:
        """
        Remove padding to revert images and masks to original size.
        Inputs:
        - original_px: int, number of pixels in each dimension
        """
        # Loop over all image paths, remove padding
        for file_path in tqdm(self.img_paths, desc="Removing padding from images"):
            current_img = rioxarray.open_rasterio(str(file_path))

            # Get the current dimensions
            current_height = current_img.rio.height
            current_width = current_img.rio.width

            if current_height > original_px or current_width > original_px:
                # Keep only the bottom-left corner with original_px dimensions
                # This removes padding from top and right sides
                current_img = current_img.isel(
                    y=slice(current_height - original_px, current_height),
                    x=slice(0, original_px),
                )
                current_img.rio.to_raster(file_path)

        # Loop over all mask paths, remove padding
        for file_path in tqdm(self.mask_paths, desc="Removing padding from masks"):
            current_mask = np.load(file_path)

            if len(current_mask.shape) == 3:  # Shape is (C, H, W)
                # Keep only the bottom-left corner, removing padding from top and right
                h = current_mask.shape[1]
                current_mask = current_mask[:, h - original_px : h, :original_px]
            else:  # Shape is (H, W)
                # Keep only the bottom-left corner, removing padding from top and right
                h = current_mask.shape[0]
                current_mask = current_mask[h - original_px : h, :original_px]

            np.save(file_path, current_mask)

    def _compute_weights(self) -> list:
        """
        Compute weights for each image based on its filename.
        If file is named with w1, weight is 1, if w2, weight is 2.
        """

        # Loop over all image paths, if w1 in filename, append 1, else append 2
        weights = []
        for file_path in self.img_paths:
            if "w1" in file_path.stem.split("_")[-1]:
                weights.append(1)
            elif "w2" in file_path.stem.split("_")[-1]:
                weights.append(2)
            else:
                raise ValueError(f"Filename {file_path.name} does not contain w1 or w2")

        return weights

    def apply_mask_type(
        self,
        mask_type: str = "dilated",
        dilation_size: int = 10,
        gaussian_sigma: int = 5,
        inverted: bool = True,
    ) -> None:
        """
        Apply a mask type to the original mask with single pixel width levees.

        Inputs:
        - mask_type: str, type of mask to apply. Choose from 'dilated' or 'gaussian'.
        - dilation_size: int, size of the dilation kernel.
        - gaussian_sigma: int, standard deviation of the gaussian kernel.
        - inverted: bool, whether the mask is inverted or not, if target pixels are 0, set to True.

        Outputs:
        - None: save the modified masks back to the original paths.
        """

        # Warn the user about data overwriting
        user_confirmation = input(
            f"WARNING: This operation will overwrite the original mask files in {self.masks_dir}. "
            f"Ensure you have saved a backup if needed. "
            f"Do you want to proceed? (yes/no): "
        )

        if user_confirmation.lower() != "yes":
            print("Operation aborted by user.")
            return

        # we will loop over masks, for each apply the mask_type
        for current_mask_file in tqdm(self.mask_paths, desc="Processing masks"):

            # get path and load the mask, note it is in (1, H, W)
            # which is the accepted format by the apply_mask_type function
            current_mask = np.load(current_mask_file)

            current_mask = apply_mask_type(
                mask=current_mask,
                mask_type=mask_type,
                dilation_size=dilation_size,
                gaussian_sigma=gaussian_sigma,
                inverted=inverted,
            )

            # Save the modified mask back to the original path.
            np.save(current_mask_file, current_mask)

    def invert_masks(self) -> None:
        """
        Invert the masks in the dataset.
        This will change 0 to 1 and 1 to 0 in the masks.
        """

        # Warn the user about data overwriting
        user_confirmation = input(
            f"WARNING: This operation will overwrite the original mask files in {self.masks_dir}. "
            f"Ensure you have saved a backup if needed. "
            f"Do you want to proceed? (yes/no): "
        )

        if user_confirmation.lower() != "yes":
            print("Operation aborted by user.")
            return

        # Loop over all mask paths, invert the masks
        for current_mask_file in tqdm(self.mask_paths, desc="Inverting masks"):
            current_mask = np.load(current_mask_file)
            current_mask = 1 - current_mask
            np.save(current_mask_file, current_mask)

    @property
    def file_ids(self) -> list:
        """
        Get the file IDs from the image paths.

        Eg for image path:
        /share/gpu5/pmucha/fathom/levee-hunter/data/processed/1m_1024/images/USGS_1M_10_x51y506_OR_OLCMetro_2019_A19_p5_w1.tif

        The file ID will be USGS_1M_10_x51y506_OR_OLCMetro_2019_A19 (always before _p{index})
        """

        # if not weighted, it might not end with w1 or w2
        # but we check, (assuming all images are either weighted or not)
        # we check if the first one ends with w1 or w2
        # if not, then it ends with _p{index}.tif so
        # we split by _ and take all but the last element
        # if it is either weighted or ends with w1 or w2, then those ifs are not passed
        # we go to the final return and this time take all but the last two elements
        if not self.weighted:
            first_tif_file = self.img_paths[0]
            weight_suffix = ["w1", "w2"]
            if not first_tif_file.stem.split("_")[-1] in weight_suffix:
                return [
                    "_".join(img_path.stem.split("_")[:-1])
                    for img_path in self.img_paths
                ]
        return ["_".join(img_path.stem.split("_")[:-2]) for img_path in self.img_paths]

    @overload
    def __load_image(
        self, idx: int, values_only: True
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    @overload
    def __load_image(
        self, idx: int, values_only: False
    ) -> Tuple[xr.DataArray, np.ndarray]: ...

    def __load_image(self, idx: int, values_only: bool = True):
        """
        Simply load in image and its mask.
        Since we sorted paths in __init__, we can just use idx to get the right image and mask.

        Inputs:
        - idx: int, index of the image and mask to load
        - values_only: bool, if true will return only values, if false will return rioxarray object

        Outputs:
        - current_img: image of shape (1, H, W). If values_only is False, it will be rioxarray object
        - current_mask: mask of shape (1, H, W)
        """
        current_tif_file = self.img_paths[idx]
        current_img = rioxarray.open_rasterio(str(current_tif_file))

        current_mask_file = self.mask_paths[idx]
        current_mask = np.load(current_mask_file)

        if values_only:
            current_img = current_img.values

        return current_img, current_mask

    def __perform_transform(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform transformation on image and mask.
        Inputs should be 2d, but will attempt to squeeze them if not.

        Inputs:
        - image: np.ndarray, image of shape (H, W)
        - mask: np.ndarray, mask of shape (H, W)

        Outputs:
        - aug_image: np.ndarray, transformed image
        - aug_mask: np.ndarray, transformed mask
        """

        assert self.transform is not None, f"Transformation {self.transform} is None"

        if len(image.shape) > 2:
            image = image.squeeze()

        if len(mask.shape) > 2:
            mask = mask.squeeze()

        # Perform augmentation.
        # If it is in TRANSFORMS, this eans self.transform is a string
        # If not, it might be a function so attempt that in else
        if self.transform in TRANSFORMS:
            augmentation = TRANSFORMS[self.transform]
            augmented = augmentation(image=image, mask=mask)
        else:
            augmented = self.transform(image=image, mask=mask)

        aug_image = augmented["image"]
        aug_target = augmented["mask"]

        return aug_image, aug_target

    def __single_plot(
        self, idx: int, figsize: tuple, cmap: str, transform: bool
    ) -> None:
        """
        Plot a single image and its mask.
        """

        # load in the data
        current_img, current_mask = self.__load_image(idx, values_only=True)

        # They are loaded in as (1, H, W), so we need to remove the first dimension
        current_img = current_img.squeeze()
        current_mask = current_mask.squeeze()

        # If transform is true, perform self.transform on the image and target
        if transform:
            assert (
                self.transform is not None
            ), "Transform is set to True but no transform function is provided."

            current_img, current_mask = self.__perform_transform(
                image=current_img, mask=current_mask
            )

        fig, axes = plt.subplots(1, 2, figsize=figsize)  # 1 row, 2 columns

        # Plot the image
        im = axes[0].imshow(current_img, cmap=cmap)
        axes[0].set_title("Lidar Image")
        axes[0].axis("off")
        cbar = fig.colorbar(
            im, ax=axes[0], orientation="vertical", fraction=0.046, pad=0.04
        )
        cbar.set_label("Value")

        axes[1].imshow(
            current_mask, cmap=ListedColormap(["white", "black"]), vmin=0, vmax=1
        )
        axes[1].set_title("Target Image")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    def plot(
        self,
        idx: int = 0,
        figsize: tuple = (6, 3),
        cmap: str = "viridis",
        transform: bool = False,
    ) -> None:
        """
        Plot image and its mask. If idx is a list, plot all images in the list.
        """
        if isinstance(idx, list):
            for i in idx:
                self.__single_plot(
                    idx=i, figsize=figsize, cmap=cmap, transform=transform
                )
        else:
            self.__single_plot(idx=idx, figsize=figsize, cmap=cmap, transform=transform)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given input idx, return image and mask both (1, H, W).

        Inputs:
        - idx: int, index of the image and mask to load

        Outputs:
        - current_img: image of shape (1, H, W)
        - current_mask: mask of shape (1, H, W)
        """

        # They are loaded in as (1, H, W), for augmentations we will flatten them
        current_img, current_mask = self.__load_image(idx, values_only=True)

        if self.transform:
            current_img = current_img.squeeze()
            current_mask = current_mask.squeeze()

            current_img, current_mask = self.__perform_transform(
                image=current_img, mask=current_mask
            )

            # after transformation, we reshape them back to (1, H, W)
            current_img = current_img.reshape(1, *current_img.shape)
            current_mask = current_mask.reshape(1, *current_mask.shape)

        # For training we need the image and mask as torch tensors
        # Must be float for loss calculation
        current_img = torch.as_tensor(current_img, dtype=torch.float32)
        current_mask = torch.as_tensor(current_mask, dtype=torch.float32)

        if self.weighted:
            if self.weights is None:
                raise ValueError(
                    "Weights are not provided, but weighted is set to True"
                )

            current_weight = self.weights[idx]
            return current_img, current_mask, current_weight

        else:
            return current_img, current_mask

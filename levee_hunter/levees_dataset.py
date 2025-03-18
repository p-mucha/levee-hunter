from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import rioxarray
import torch
from torch.utils.data import Dataset
from typing import Tuple
import warnings

from levee_hunter.augmentations import TRANSFORMS


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

    def _compute_weights(self) -> list:
        """
        Compute weights for each image based on its filename.
        If file is named with w1, weight is 1, if w2, weight is 2.
        """

        # Loop over all image paths, if w1 in filename, append 1, else append 2
        weights = []
        for file_path in self.img_paths:
            if "w1" in file_path.name:
                weights.append(1)
            elif "w2" in file_path.name:
                weights.append(2)
            else:
                raise ValueError(f"Filename {file_path.name} does not contain w1 or w2")

        return weights

    def __load_image(
        self, idx: int, values_only: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def __perform_transform(self, image, mask):
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

        if len(image.shape) == 3:
            image = image.squeeze()

        if len(mask.shape) == 3:
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
        # Not providing dtype on purpose, so that torch will base it on the input
        current_img = torch.as_tensor(current_img)
        current_mask = torch.as_tensor(current_mask)

        if self.weighted:
            if self.weights is None:
                raise ValueError(
                    "Weights are not provided, but weighted is set to True"
                )

            current_weight = self.weights[idx]
            return current_img, current_mask, current_weight

        else:
            return current_img, current_mask

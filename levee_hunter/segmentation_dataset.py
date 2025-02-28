from torch.utils.data import Dataset
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from scipy.ndimage import binary_dilation, gaussian_filter
import warnings

import levee_hunter.augmentations as lhAug


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images,
        targets,
        transform=None,
        split=False,
        patch_size=None,
        final_size=None,
        overlap=0,
        weights=None,
        file_ids=None,
    ):
        if split:
            if patch_size is None or final_size is None:
                raise ValueError(
                    "Error: `patch_size` and `final_size` must be provided when `split=True`."
                )
            self.images, self.targets = self.split_and_pad(
                images, targets, patch_size, final_size, overlap
            )
        else:
            if patch_size is not None or final_size is not None:
                warnings.warn(
                    "Warning: `patch_size` and `final_size` are provided but `split=False`. "
                    "These parameters will be ignored."
                )
            self.images = images
            self.targets = targets

        self.transform = transform
        if self.transform not in [
            "train_transform",
            "no_deformations_transform",
            "normalize_only",
        ]:
            FutureWarning(
                "transform should be a string, this will be changed in the future"
            )
        self.empty_images = None
        self.empty_targets = None

        # For weighted training
        self.weights = weights
        self.weights_return = False

        # Keep track which file each image originated from
        self.file_ids = file_ids

        self.overlap = overlap

    def split_and_pad(self, images, targets, patch_size, final_size, overlap):
        """Splits the image and mask into smaller patches and pads them."""

        smaller_images = []
        smaller_targets = []
        stride = patch_size - overlap

        # Eg for images.shape (22, 3000, 3000)
        # this will loop over 22 images (3000, 3000)
        for image_no in range(len(images)):
            image = images[image_no]
            target = targets[image_no]

            # Notice since image = images[image_no], its shape is 2d
            # Loop to cover entire image
            for i in range(0, max(image.shape[0] - patch_size, 0) + 1, stride):
                for j in range(0, max(image.shape[1] - patch_size, 0) + 1, stride):

                    # For given image:
                    # Divide the image into (patch_size, patch_size) patches
                    smaller_image = image[i : i + patch_size, j : j + patch_size]
                    smaller_target = target[i : i + patch_size, j : j + patch_size]

                    # Pad to (final_size, final_size)
                    pad_h = final_size - smaller_image.shape[0]
                    pad_w = final_size - smaller_image.shape[1]

                    if (pad_h > 0 or pad_w > 0) and patch_size == final_size:
                        warnings.warn(
                            f"Padding ({pad_h}, {pad_w}) was added, but patch_size == final_size. "
                            "This may be unintended and could affect training."
                        )

                    smaller_image = torch.tensor(smaller_image, dtype=torch.float32)
                    smaller_target = torch.tensor(smaller_target, dtype=torch.float32)

                    smaller_image = F.pad(
                        smaller_image, (0, pad_w, 0, pad_h), mode="constant", value=0
                    )
                    smaller_target = F.pad(
                        smaller_target, (0, pad_w, 0, pad_h), mode="constant", value=1
                    )

                    # Reshape to (1, 256, 256) by adding the channel dimension
                    smaller_image = smaller_image.unsqueeze(0)
                    smaller_target = smaller_target.unsqueeze(0)

                    smaller_images.append(smaller_image)
                    smaller_targets.append(smaller_target)

        return smaller_images, smaller_targets

    def remove_empty(self, keep_empty=0.0):
        """
        Separates empty images (where min value in targets == 0) and optionally
        adds back a fraction of them based on `keep_empty`.

        Parameters:
        - keep_empty (float): Fraction of empty images to add back (e.g., 0.3 adds 30% of empty images).
        """
        # Convert targets to numpy array
        targets = np.array(self.targets)

        # Find indices where min value in target is 0 (empty images)
        valid_indices = np.where(np.min(targets, axis=(1, 2, 3)) == 0)[0]

        # Get non-empty and empty images
        non_empty_images = [self.images[i] for i in valid_indices]
        empty_images = [
            self.images[i] for i in range(len(self.images)) if i not in valid_indices
        ]

        non_empty_targets = [self.targets[i] for i in valid_indices]
        empty_targets = [
            self.targets[i] for i in range(len(self.targets)) if i not in valid_indices
        ]

        # Store non-empty and empty images separately
        self.images = non_empty_images
        self.empty_images = empty_images
        self.targets = non_empty_targets
        self.empty_targets = empty_targets

        # Add back a fraction of empty images
        if keep_empty > 0.0:
            num_to_add = int(
                len(non_empty_images) * keep_empty
            )  # Calculate how many empty images to add back
            num_to_add = min(
                num_to_add, len(empty_images)
            )  # Ensure we don't exceed available empty images

            if num_to_add > 0:
                selected_indices = np.random.choice(
                    len(empty_images), size=num_to_add, replace=False
                )

                # Add selected empty images and targets back to the dataset
                self.images += [empty_images[i] for i in selected_indices]
                self.targets += [empty_targets[i] for i in selected_indices]

    def remove_invalid_images(self):
        """
        Removes images where the minimum pixel value is less than -9999,
        along with their corresponding targets.
        This might happen if data is missing in a given region.
        Basically this function allows (after splitting large image into smaller ones),
        to keep some parts of the original image even if there are parts missing on it,
        while removing any smaller images that might contain those missing parts.
        """
        images = np.array(self.images)

        # Find indices where the min value in the image is >= -9999 (valid images)
        valid_indices = np.where(np.min(images, axis=(1, 2, 3)) >= -9999)[0]

        # Keep only valid images and their corresponding targets
        self.images = [self.images[i] for i in valid_indices]
        self.targets = [self.targets[i] for i in valid_indices]

    def __perform_transform(self, image, target):
        # Handle string versions of transforms
        if self.transform == "train_transform":
            augmented = lhAug.train_transform(image=image, mask=target)
        elif self.transform == "no_deformations_transform":
            augmented = lhAug.no_deformations_transform(image=image, mask=target)
        elif self.transform == "normalize_only":
            augmented = lhAug.normalize_only(image=image, mask=target)
        else:
            augmented = self.transform(image=image, mask=target)

        return augmented["image"], augmented["mask"]

    def __apply_mask_type(
        self, target, mask_type="dilated", dilation_size=10, gaussian_sigma=5.0
    ):

        # This block handles potentially wrong option selection by user
        # Unlike get_mask, there is no need to accept None as input here
        valid_mask_types = {"dilated", "gaussian"}
        if mask_type not in valid_mask_types:
            raise ValueError(
                f"Invalid mask_type. Choose one of {valid_mask_types}, or None."
            )
        if dilation_size != 10 and mask_type != "dilated":
            warnings.warn(
                "dilation_size will be ignored if mask_type is not 'dilated'."
            )
        if gaussian_sigma != 5.0 and mask_type != "gaussian":
            warnings.warn(
                "gaussian_sigma will be ignored if mask_type is not 'gaussian'."
            )

        # This block handles different types of the target
        to_tensor = False
        if isinstance(target, torch.Tensor):
            target = target.numpy()
            to_tensor = True

        # Sometimes our self.targets may have 4d shape like
        # (123, 1, 512, 512), then if one tries this on self.targets[ix]
        # we need to squeeze
        unsqueeze = False
        if len(target.shape) == 3:
            target = target.squeeze()
            unsqueeze = True

        if mask_type == "dilated":
            # Apply binary dilation
            structure = np.ones((dilation_size, dilation_size), dtype=bool)
            target = binary_dilation(target, structure=structure).astype(np.uint8)
        elif mask_type == "gaussian":
            # Apply Gaussian filter
            target = gaussian_filter(target.astype(float), sigma=gaussian_sigma)
            target = (target > 0.1).astype(np.uint8)

        if unsqueeze:
            target = target.reshape(1, *target.shape)

        if to_tensor:
            target = torch.tensor(target, dtype=torch.float32)

        return target

    def __single_plot(
        self, idx, transform, figsize, mask_type, dilation_size, gaussian_sigma
    ):
        fig, axes = plt.subplots(1, 2, figsize=figsize)  # 1 row, 2 columns

        # Regardles of whether self.to_array() has been used or not
        # images[idx] will have shape (1, N, N), so we squeeze it for plotting
        image = np.array(self.images[idx].squeeze())
        target = np.array(self.targets[idx].squeeze())

        if mask_type is not None:
            target = self.__apply_mask_type(
                target,
                mask_type=mask_type,
                dilation_size=dilation_size,
                gaussian_sigma=gaussian_sigma,
            )

        # If transform is true, perform self.transform on the image and target
        if transform:
            assert (
                self.transform is not None
            ), "Transform is set to True but no transform function is provided."

            image, target = self.__perform_transform(image, target)

        im = axes[0].imshow(image, cmap="viridis")
        axes[0].set_title("Lidar Image")
        axes[0].axis("off")

        cbar = fig.colorbar(
            im, ax=axes[0], orientation="vertical", fraction=0.046, pad=0.04
        )
        cbar.set_label("Value")

        axes[1].imshow(target, cmap=ListedColormap(["white", "black"]), vmin=0, vmax=1)
        axes[1].set_title("Target Image")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    def plot(
        self,
        idx=0,
        transform=False,
        figsize=(6, 3),
        mask_type=None,
        dilation_size=10,
        gaussian_sigma=5.0,
    ):
        if isinstance(idx, list) or isinstance(idx, np.ndarray):
            for i in idx:
                self.__single_plot(
                    i, transform, figsize, mask_type, dilation_size, gaussian_sigma
                )
        else:
            self.__single_plot(
                idx, transform, figsize, mask_type, dilation_size, gaussian_sigma
            )

    def to_array(self):
        """
        Converts the images and targets to numpy arrays.
        """
        self.images = np.array(self.images)
        self.targets = np.array(self.targets)

    @property
    def is_array(self):
        return isinstance(self.images, np.ndarray) and isinstance(
            self.targets, np.ndarray
        )

    @property
    def shape(self):
        return np.array(self.images).shape

    def __add__(self, other):
        """
        Concatenates two datasets.
        """
        if not isinstance(other, SegmentationDataset):
            raise ValueError("Error: Can only concatenate SegmentationDataset objects.")

        # Check if weights are consistent
        if self.weights is not None and other.weights is not None:
            if isinstance(self.weights, np.ndarray) and isinstance(
                other.weights, np.ndarray
            ):
                self.weights = np.concatenate((self.weights, other.weights), axis=0)

            elif isinstance(self.weights, torch.Tensor) and isinstance(
                other.weights, torch.Tensor
            ):
                self.weights = torch.cat((self.weights, other.weights), dim=0)

            elif isinstance(self.weights, list) and isinstance(other.weights, list):
                self.weights += other.weights

            else:
                raise TypeError(
                    f"Error: Mismatched weight types ({type(self.weights)} vs {type(other.weights)})."
                )

        # give error if weights are provided for one dataset but not the other
        elif (
            self.weights is not None
            and other.weights is None
            or self.weights is None
            and other.weights is not None
        ):
            raise ValueError(
                "Error: Weights are provided for one dataset but not the other."
            )

        # Check if transforms are consistent
        # Give error if transforms are not the same
        if self.transform != other.transform:
            raise ValueError("Error: Transforms are different for the two datasets.")

        # Check if file_ids are consistent
        # Give error if one dataset has file_ids and the other doesn't
        if (
            self.file_ids is None
            and other.file_ids is not None
            or self.file_ids is not None
            and other.file_ids is None
        ):
            raise ValueError(
                "Error: `file_ids` are provided for one dataset but not the other."
            )

        # This is kind of simplified, but lets assume both have the same type
        # If they are numpy arrays, just concatenate them
        # If they are lists, then add them
        if self.file_ids is not None and other.file_ids is not None:
            if isinstance(self.file_ids, np.ndarray) and isinstance(
                other.file_ids, np.ndarray
            ):
                self.file_ids = np.concatenate((self.file_ids, other.file_ids), axis=0)

            else:
                self.file_ids = self.file_ids + other.file_ids

        # Handle images now, if they are numpy arrays, just concatenate
        # If they are not, then try to add them, as they should be lists
        # Additionally, we will assume targets are the same type as images
        # Otherwise something weird has happened
        if isinstance(self.images, np.ndarray) and isinstance(other.images, np.ndarray):
            if len(self.images.shape) == 2:
                self.images = self.images.reshape(1, *self.images.shape)
                self.targets = self.targets.reshape(1, *self.targets.shape)

            if len(other.images.shape) == 2:
                other.images = other.images.reshape(1, *other.images.shape)
                other.targets = other.targets.reshape(1, *other.targets.shape)

            self.images = np.concatenate((self.images, other.images), axis=0)
            self.targets = np.concatenate((self.targets, other.targets), axis=0)

        else:
            self.images = self.images + other.images
            self.targets = self.targets + other.targets

        return self

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Augmentations works on numpy arrays
        # So to use this option first convert to numpy arrays
        # But then we want tensors of shape (1, N, N) as outputs
        # So convert back to tensors and reshape
        if self.transform and not self.is_array:
            image = np.array(self.images[idx])
            target = np.array(self.targets[idx])

        else:
            image = self.images[idx]
            target = self.targets[idx]

        if self.transform:
            N = image.shape[1]

            image, target = self.__perform_transform(image, target)

            image = image.reshape(1, N, N)
            target = target.reshape(1, N, N)

            image = torch.tensor(image, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)

        assert (
            image.shape[1] == image.shape[2]
        ), f"Image is not square. Shape: {image.shape}"
        assert image.shape == target.shape, "Image and target shapes do not match."

        if self.weights_return:
            if self.weights is None:
                raise ValueError(
                    "Error: `return_weights=True` but `weights` is None. "
                    "Provide a valid weights array when initializing the dataset."
                )
            return image, target, self.weights[idx]

        return image, target

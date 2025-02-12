from torch.utils.data import Dataset
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images,
        targets,
        transform=None,
        split=False,
        patch_size=250,
        final_size=256,
        overlap=0,
    ):
        if split:
            self.images, self.targets = self.split_and_pad(
                images, targets, patch_size, final_size, overlap
            )
        else:
            self.images = images
            self.targets = targets

        self.transform = transform
        self.empty_images = None
        self.empty_targets = None

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

    def __single_plot(self, idx, transform, figsize):

        fig, axes = plt.subplots(1, 2, figsize=figsize)  # 1 row, 2 columns

        # Regardles of whether self.to_array() has been used or not
        # images[idx] will have shape (1, N, N), so we squeeze it for plotting
        image = np.array(self.images[idx].squeeze())
        target = np.array(self.targets[idx].squeeze())

        # If transform is true, perform self.transform on the image and target
        if transform:
            assert (
                self.transform is not None
            ), "Transform is set to True but no transform function is provided."

            augmented = self.transform(image=image, mask=target)
            image = augmented["image"]
            target = augmented["mask"]

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

    def plot(self, idx=0, transform=False, figsize=(6, 3)):
        if isinstance(idx, list) or isinstance(idx, np.ndarray):
            for i in idx:
                self.__single_plot(i, transform, figsize)
        else:
            self.__single_plot(idx, transform, figsize)

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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Augmentations works on numpy arrays
        # So to use this option first convert to numpy arrays
        # But then we want tensors of shape (1, N, N) as outputs
        # So convert back to tensors and reshape
        if self.transform and not self.is_array:
            self.to_array()

        image = self.images[idx]
        target = self.targets[idx]

        if self.transform:
            N = image.shape[1]
            augmented = self.transform(image=image, mask=target)
            image = augmented["image"]
            target = augmented["mask"]

            image = image.reshape(1, N, N)
            target = target.reshape(1, N, N)

            image = torch.tensor(image, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)

        assert (
            image.shape[1] == image.shape[2]
        ), f"Image is not square. Shape: {image.shape}"
        assert image.shape == target.shape, "Image and target shapes do not match."

        return image, target

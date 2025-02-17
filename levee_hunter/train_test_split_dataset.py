from sklearn.model_selection import train_test_split
import numpy as np
import warnings
from levee_hunter.segmentation_dataset import SegmentationDataset
from levee_hunter.augmentations import normalize_only


def train_test_split_dataset(dataset, test_size=0.2):
    """
    Splits a SegmentationDataset into training and validation datasets.

    Args:
        dataset (SegmentationDataset): The dataset to split.
        test_size (float): Proportion of the dataset to use for validation.

    Returns:
        training_dataset (SegmentationDataset): Training dataset.
        val_dataset (SegmentationDataset): Validation dataset.
    """

    if not isinstance(dataset, SegmentationDataset):
        raise ValueError("Input dataset must be a SegmentationDataset instance")

    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices, test_size=test_size, random_state=42, shuffle=True
    )

    # Select corresponding images and targets
    train_images, val_images = dataset.images[train_idx], dataset.images[val_idx]
    train_targets, val_targets = (
        dataset.targets[train_idx],
        dataset.targets[val_idx],
    )

    # Right now those are lists of tensors,
    # but SegmentationDataset expects numpy arrays
    # So we need to convert them
    train_images = np.array(train_images)
    val_images = np.array(val_images)

    training_dataset = SegmentationDataset(
        images=train_images,
        targets=train_targets,
        transform=dataset.transform,
        split=False,
    )

    # Check if the dataset transform is None and warn the user
    if dataset.transform is None:
        warnings.warn(
            "Warning: The input dataset has `transform=None`. "
            "The training dataset therefore also has no transformatons currently."
        )

    val_dataset = SegmentationDataset(
        images=val_images, targets=val_targets, transform=normalize_only, split=False
    )

    # Also set the weights for the training dataset, if they exist
    # Otherwise they will be none by default.
    weights = dataset.weights
    if weights is not None:
        training_dataset.weights = weights[train_idx]
        val_dataset.weights = weights[val_idx]

    return training_dataset, val_dataset

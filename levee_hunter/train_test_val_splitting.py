import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

from levee_hunter.levees_dataset import LeveesDataset


def validation_split(
    dataset: LeveesDataset, val_percentage: float = None, val_file_ids: list = None, seed: int = 42,
) -> Tuple[LeveesDataset, LeveesDataset]:
    """
    Split the dataset into training and validation sets based on a percentage or specific file IDs,
    ensuring that all parts of the same image area are kept in the same set.

    Inputs:
    - dataset: LeveesDataset object to split.
    - val_percentage: float, percentage of the dataset to use for validation.
    - val_file_ids: list, specific file IDs to use for validation.
    - seed: int, random seed for reproducibility.

    Outputs:
    - train_dataset: LeveesDataset object, with the images not in the validation set.
    - val_dataset: LeveesDataset object, with the images in the validation set.
    """
    if not isinstance(dataset, LeveesDataset):
        raise ValueError("Input dataset must be a LeveesDataset instance")

    if val_percentage is None and val_file_ids is None:
        raise ValueError("Either val_percentage or val_file_ids must be provided")

    if val_percentage is not None and not (0.0 < val_percentage < 1.0):
        raise ValueError("val_percentage must be between 0 and 1")

    # Extract unique image area identifiers (everything before the patch designation)
    file_ids = sorted(dataset.file_ids)
    unique_areas = set()
    for file_id in file_ids:
        # Split at the last underscore to separate the area identifier from the patch number
        parts = file_id.rsplit('_', 1)
        if len(parts) > 1:
            unique_areas.add(parts[0])

    # Determine validation areas either from percentage or explicit file_ids
    val_areas = set()
    if val_file_ids is not None:
        # Use explicit file IDs
        for file_id in val_file_ids:
            if '_p' in file_id:
                # If it's a patch ID, extract the area part
                val_areas.add(file_id.rsplit('_', 1)[0])
            else:
                # Otherwise assume it's already an area ID
                val_areas.add(file_id)
    else:
        # Use percentage-based split
        unique_areas = sorted(unique_areas)
        np.random.seed(seed)
        np.random.shuffle(unique_areas)
        split_idx = int(len(unique_areas) * (1 - val_percentage))
        val_areas = set(unique_areas[split_idx:])
    
    # Train areas are all areas not in validation
    train_areas = set(unique_areas) - val_areas

    # Get indices for train and validation sets
    indices_to_keep = []
    indices_to_move = []
    
    for i, file_id in enumerate(dataset.file_ids):
        area_id = file_id.rsplit('_', 1)[0]
        if area_id in train_areas:
            indices_to_keep.append(i)
        elif area_id in val_areas:
            indices_to_move.append(i)

    # ----------------- Handle the validation dataset -----------------
    validation_dataset = LeveesDataset(
        images_dir=dataset.images_dir,
        masks_dir=dataset.masks_dir,
        transform="normalize_only",
        weighted=dataset.weighted,
    )
    validation_dataset.img_paths = [
        validation_dataset.img_paths[i] for i in indices_to_move
    ]
    validation_dataset.mask_paths = [
        validation_dataset.mask_paths[i] for i in indices_to_move
    ]

    if validation_dataset.weighted:
        validation_dataset.weights = [
            validation_dataset.weights[i] for i in indices_to_move
        ]

    # ----------------- Handle the train dataset -----------------
    train_dataset = LeveesDataset(
        images_dir=dataset.images_dir,
        masks_dir=dataset.masks_dir,
        transform=dataset.transform,
        weighted=dataset.weighted,
    )
    train_dataset.img_paths = [
        train_dataset.img_paths[i] for i in indices_to_keep
    ]
    train_dataset.mask_paths = [
        train_dataset.mask_paths[i] for i in indices_to_keep
    ]

    if train_dataset.weighted:
        train_dataset.weights = [
            train_dataset.weights[i] for i in indices_to_keep
        ]

    return train_dataset, validation_dataset


def train_test_split_dataset(dataset, test_size=0.2):
    """
    Splits a LeveesDataset into training and validation datasets.

    Inputs:
    - dataset (LeveesDataset): The dataset to split.
    - test_size (float): Proportion of the dataset to use for validation.

    Returns:
        dataset (LeveesDataset): Training dataset.
        val_dataset (LeveesDataset): Validation dataset.
    """

    if not isinstance(dataset, LeveesDataset):
        raise ValueError("Input dataset must be a LeveesDataset instance")

    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=42, shuffle=True
    )

    # ----------------- Handle the test dataset -----------------
    # create a test dataset object based on the original one
    test_dataset = LeveesDataset(
        images_dir=dataset.images_dir,
        masks_dir=dataset.masks_dir,
        transform="normalize_only",
        weighted=dataset.weighted,
    )
    # The images, masks and weights are the ones that are at indices indices_to_move
    test_dataset.img_paths = [test_dataset.img_paths[i] for i in test_idx]
    test_dataset.mask_paths = [test_dataset.mask_paths[i] for i in test_idx]

    if test_dataset.weighted:
        test_dataset.weights = [test_dataset.weights[i] for i in test_idx]

    # ----------------- Handle the train dataset -----------------
    train_dataset = LeveesDataset(
        images_dir=dataset.images_dir,
        masks_dir=dataset.masks_dir,
        transform="train_transform",
        weighted=dataset.weighted,
    )
    # We don't want to double the images, so we remove the images that we moved to the validation set
    train_dataset.img_paths = [train_dataset.img_paths[i] for i in train_idx]
    train_dataset.mask_paths = [train_dataset.mask_paths[i] for i in train_idx]

    if train_dataset.weighted:
        train_dataset.weights = [train_dataset.weights[i] for i in train_idx]

    return train_dataset, test_dataset

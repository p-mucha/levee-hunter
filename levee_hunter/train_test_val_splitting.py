import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

from levee_hunter.levees_dataset import LeveesDataset


def validation_split(
    dataset: LeveesDataset, file_ids: list
) -> Tuple[LeveesDataset, LeveesDataset]:
    """
    Take original dataset and based on file_ids provided, move those images to validation set.

    Inputs:
    - dataset: LeveesDataset object, from which we will move the images
    - file_ids: list of file ids to move to validation set

    Outputs:
    - train_dataset: LeveesDataset object, with the images not in file_ids
    - val_dataset: LeveesDataset object, with the images in file_ids
    """
    if not isinstance(dataset, LeveesDataset):
        raise ValueError("Input dataset must be a SegmentationDataset instance")

    # Look at dataset.file_ids from the dataset, for each ID we can check if
    # it is in the provided list of file_ids. We get indices of those dataset.file_ids
    # that are in the provided list. Those indices tell us which elements to move
    # to the validation set. The rest will be kept in the train_test dataset.
    all_file_ids = dataset.file_ids
    indices_to_move = np.where(np.isin(all_file_ids, file_ids))[0]
    indices_to_keep = np.where(~np.isin(all_file_ids, file_ids))[0]

    # ----------------- Handle the validation dataset -----------------
    # create a validation dataset object based on the original one
    validation_dataset = LeveesDataset(
        images_dir=dataset.images_dir,
        masks_dir=dataset.masks_dir,
        transform="normalize_only",
        weighted=dataset.weighted,
    )
    # The images, masks and weights are the ones that are at indices indices_to_move
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

    # ----------------- Handle the train_test dataset -----------------
    # create a train_test dataset object based on the original one
    train_test_dataset = LeveesDataset(
        images_dir=dataset.images_dir,
        masks_dir=dataset.masks_dir,
        transform=dataset.transform,
        weighted=dataset.weighted,
    )
    # We don't want to double the images, so we remove the images that we moved to the validation set
    train_test_dataset.img_paths = [
        train_test_dataset.img_paths[i] for i in indices_to_keep
    ]
    train_test_dataset.mask_paths = [
        train_test_dataset.mask_paths[i] for i in indices_to_keep
    ]

    if train_test_dataset.weighted:
        train_test_dataset.weights = [
            train_test_dataset.weights[i] for i in indices_to_keep
        ]

    return train_test_dataset, validation_dataset


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

import albumentations as A

# Note:
# When new transformation is added, it should be added to the TRANSFORMS dictionary below
# Note: augmentations work only with 2D numpy arrays, not with Torch tensors

# Define augmentations
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.VerticalFlip(p=0.5),  # Random vertical flip
        A.RandomRotate90(p=0.5),  # Random 90-degree rotation
        A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
        ),  # Small shifts, scaling, rotation
        A.ElasticTransform(
            alpha=1, sigma=50, alpha_affine=50, p=0.5
        ),  # Elastic deformation
        A.Lambda(
            name="z_score_normalize",
            image=lambda x, **kwargs: (x - x.mean()) / (x.std() + 1e-7),
        ),
        # A.Lambda(image=lambda x, **kwargs: (x - x.min()) / (x.max() - x.min() + 1e-7)),
    ]
)

no_deformations_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.VerticalFlip(p=0.5),  # Random vertical flip
        A.RandomRotate90(p=0.5),  # Random 90-degree rotation
        A.Lambda(
            name="z_score_normalize",
            image=lambda x, **kwargs: (x - x.mean()) / (x.std() + 1e-7),
        ),
    ]
)

divide_by255_normalize = A.Compose(
    [
        A.Normalize(mean=0.0, std=1.0),
    ]
)

z_score_normalize = A.Compose(
    [
        A.Lambda(
            name="z_score_normalize",
            image=lambda x, **kwargs: (x - x.mean()) / (x.std() + 1e-7),
        )
    ]
)

min_max_normalize = A.Compose(
    [
        A.Lambda(
            name="min_max_normalize",
            image=lambda x, **kwargs: (x - x.min()) / (x.max() - x.min() + 1e-7),
        )
    ]
)

normalize_only = A.Compose(
    # [A.Lambda(image=lambda x, **kwargs: (x - x.min()) / (x.max() - x.min() + 1e-7))]
    [
        A.Lambda(
            name="normalize_only",
            image=lambda x, **kwargs: (x - x.mean()) / (x.std() + 1e-7),
        )
    ]
)


# Create a dictionary mapping names to transforms
TRANSFORMS = {
    "train_transform": train_transform,
    "no_deformations_transform": no_deformations_transform,
    "divide_by255_normalize": divide_by255_normalize,
    "z_score_normalize": z_score_normalize,
    "min_max_normalize": min_max_normalize,
    "normalize_only": normalize_only,
}

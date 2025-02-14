import albumentations as A

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
        A.Normalize(mean=0.0, std=1.0),  # Normalize to 0 mean and 1 std
    ]
)

no_deformations_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.VerticalFlip(p=0.5),  # Random vertical flip
        A.RandomRotate90(p=0.5),  # Random 90-degree rotation
        A.Normalize(mean=0.0, std=1.0),  # Normalize to 0 mean and 1 std
    ]
)

normalize_only = A.Compose(
    [
        A.Normalize(mean=0.0, std=1.0),  # Normalize to 0 mean and 1 std
    ]
)

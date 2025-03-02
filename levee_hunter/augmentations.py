import albumentations as A

# Important Note:
# When you add new transform, it also should be added to segmentation_dataset
# in the __init__ method and in __perform_transform method.


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
        A.Lambda(image=lambda x, **kwargs: (x - x.mean()) / (x.std() + 1e-7))
    ]
)

no_deformations_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.VerticalFlip(p=0.5),  # Random vertical flip
        A.RandomRotate90(p=0.5),  # Random 90-degree rotation
        A.Lambda(image=lambda x, **kwargs: (x - x.mean()) / (x.std() + 1e-7))
    ]
)

divide_by255_normalize = A.Compose(
    [
        A.Normalize(mean=0.0, std=1.0), 
    ]
)

z_score_normalize = A.Compose([
    A.Lambda(image=lambda x, **kwargs: (x - x.mean()) / (x.std() + 1e-7))
])

min_max_normalize = A.Compose([
    A.Lambda(image=lambda x, **kwargs: (x - x.mean()) / (x.std() + 1e-7))
])
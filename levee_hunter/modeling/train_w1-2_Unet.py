import torch
from torch.utils.data import DataLoader
from levee_hunter.plots import infer_and_visualize
from levee_hunter.augmentations import train_transform, normalize_only
from levee_hunter.segmentation_dataset import SegmentationDataset
import numpy as np
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def load_datasets():
    # Load datasets
    good_dataset = torch.load(
        "../../data/processed/w1-2-based-datasets/good_dataset_256_nooverlap.pth"
    )
    bad_dataset = torch.load(
        "../../data/processed/w1-2-based-datasets/bad_dataset_256_nooverlap.pth"
    )

    good_dataset.remove_empty(keep_empty=0.0)
    bad_dataset.remove_empty(keep_empty=0.0)

    # Set transformations
    good_dataset.transform = train_transform
    bad_dataset.transform = normalize_only

    # Select challenging images for validation
    by_eye_val_images = np.array(bad_dataset.images)[
        [49, 53, 85, 4, 11, 26, 52, 58, 70, 79, 106, 110, 121]
    ]
    by_eye_val_targets = np.array(bad_dataset.targets)[
        [49, 53, 85, 4, 11, 26, 52, 58, 70, 79, 106, 110, 121]
    ]

    val_by_eye_dataset = SegmentationDataset(
        images=by_eye_val_images,
        targets=by_eye_val_targets,
        transform=normalize_only,
        patch_size=256,
        final_size=256,
        overlap=0,
    )

    # Prepare validation dataset
    good_for_val_images = np.delete(np.array(bad_dataset.images), [49, 53, 85], axis=0)
    good_for_val_targets = np.delete(
        np.array(bad_dataset.targets), [49, 53, 85], axis=0
    )

    empty_images = np.array(good_dataset.empty_images)[:20]
    empty_targets = np.array(good_dataset.empty_targets)[:20]

    val_dataset = SegmentationDataset(
        images=np.concatenate([good_for_val_images, empty_images]).squeeze(),
        targets=np.concatenate([good_for_val_targets, empty_targets]).squeeze(),
        transform=normalize_only,
        split=False,
    )

    return good_dataset, val_dataset


def train_model(
    epochs=3, batch_size=32, save_path="../../models/w1-2-based-model/unet-model2.pth"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    good_dataset, val_dataset = load_datasets()

    train_loader = DataLoader(good_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    # Model setup
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    train_loss_list, val_loss_list = [], []

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, mask in train_loader:
            images, mask = images.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, mask in val_loader:
                images, mask = images.to(device), mask.to(device)
                output = model(images)
                loss = criterion(output, mask)
                val_loss += loss.item()

        # Compute average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(
            f"Epoch: {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}"
        )

    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"Saved model at {save_path}")

    # Plot training and validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_loss_list, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_loss_list, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------------------- Inference on Index 5 ----------------------
    print("Running inference on index 5...")
    val_dataset.transform = normalize_only  # Ensure dataset uses normalization

    sample_image, sample_mask = val_dataset[5]  # Select index 5
    output = infer_and_visualize(model, sample_image, sample_mask, apply_sigmoid=True)

    print("Inference completed for index 5.")


if __name__ == "__main__":
    train_model()

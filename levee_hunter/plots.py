import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def infer_and_visualize(model, image_tensor, mask_tensor, apply_sigmoid=True):

    model.eval()

    if apply_sigmoid:
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))
            output = torch.sigmoid(output).cpu().squeeze().numpy()
            output = (output > 0.5).astype(np.uint8)
    else:
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))
            output = output.squeeze()
            output = np.array(output)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image_tensor.squeeze())

    plt.subplot(1, 3, 2)
    plt.title("Prediction")
    plt.imshow(output, cmap=ListedColormap(["white", "black"]), vmin=0, vmax=1)

    plt.subplot(1, 3, 3)
    plt.title("Target Mask")
    plt.imshow(
        mask_tensor.squeeze(),
        cmap=ListedColormap(["white", "black"]),
        vmin=0,
        vmax=1,
    )

    plt.show()

    return output


def plot_img_and_target(img, target, figsize=(12, 6)):
    if img.shape[0] == 1:
        img = img.squeeze()

    if target.shape[0] == 1:
        target = target.squeeze()

    fig, axes = plt.subplots(1, 2, figsize=figsize)  # 1 row, 2 columns

    # Plot the first image
    im = axes[0].imshow(img, cmap="viridis")
    axes[0].set_title("Lidar Image")
    axes[0].axis("off")

    cbar = fig.colorbar(
        im, ax=axes[0], orientation="vertical", fraction=0.046, pad=0.04
    )
    cbar.set_label("Value")

    # Plot the second image
    axes[1].imshow(
        target, cmap=ListedColormap(["white", "black"]), vmin=0, vmax=1
    )  # Explicitly map 0->white, 1->black
    axes[1].set_title("Target Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def plot_training_validation_loss(train_loss_list, val_loss_list):
    # Plot training and validation loss
    epochs_range = range(1, len(train_loss_list) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_loss_list, label="Train Loss")
    plt.plot(epochs_range, val_loss_list, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

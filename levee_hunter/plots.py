import torch
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from levee_hunter.modeling.inference import infer


def infer_and_visualize_old(
    model,
    image_tensor,
    mask_tensor,
    apply_sigmoid=True,
    threshold=0.5,
    device=None,
    figsize=(12, 4),
):

    model.eval()

    # Convert image to tensor if it is not already
    if not isinstance(image_tensor, torch.Tensor):
        image_tensor = torch.Tensor(image_tensor)

    # Convert image to tensor if it is not already
    if not isinstance(mask_tensor, torch.Tensor):
        mask_tensor = torch.Tensor(mask_tensor)

    if device is not None:
        image_tensor = image_tensor.to(device)

    if apply_sigmoid:
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))
            output = torch.sigmoid(output).cpu().squeeze().numpy()
            output = (output > threshold).astype(np.uint8)
    else:
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))
            output = output.squeeze().cpu().numpy()

    # Original Image
    # At this point this is tensor of shape (1, N, N)
    # Squeeze the tensor
    # Also it might be on cuda, for plotting we move to cpu
    # Otherwise, matplotlib will throw an error
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)

    # Left image
    im0 = ax0.imshow(image_tensor.cpu().squeeze(), cmap="viridis")
    ax0.set_title("Original Image")
    ax0.axis("off")

    # Prediction
    ax1.imshow(output, cmap=ListedColormap(["white", "black"]), vmin=0, vmax=1)
    ax1.set_title("Prediction")
    ax1.axis("off")

    # Target Mask
    ax2.imshow(
        mask_tensor.cpu().squeeze(),
        cmap=ListedColormap(["white", "black"]),
        vmin=0,
        vmax=1,
    )
    ax2.set_title("Target Mask")
    ax2.axis("off")

    # -- Manually add a colorbar axis that does NOT shrink ax0 --
    # Get the position of ax0 in figure coordinates
    pos = ax0.get_position()  # (x0, y0, width, height)

    # Decide where to place the colorbar: just to the right of ax0
    x_cbar = pos.x0 + pos.width + 0.005  # 0.01 is a small gap
    y_cbar = pos.y0
    cbar_width = 0.01
    cbar_height = pos.height

    # Create a new axis for the colorbar
    cax = fig.add_axes([x_cbar, y_cbar, cbar_width, cbar_height])

    # Draw the colorbar in this new axis
    fig.colorbar(im0, cax=cax)

    plt.show()

    return output


def plot_prediction(original_img, pred_img, target_img, figsize=(12, 4)):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)

    # Original Image
    # At this point this might be a tensor of shape (1, N, N)
    # Squeeze the tensor
    # Also it might be on cuda, for plotting we move to cpu
    # Otherwise, matplotlib will throw an error
    if isinstance(original_img, torch.Tensor):
        original_img = original_img.cpu()
    im0 = ax0.imshow(original_img.squeeze(), cmap="viridis")
    ax0.set_title("Original Image")
    ax0.axis("off")

    # Prediction
    if isinstance(pred_img, torch.Tensor):
        pred_img = pred_img.cpu()
    ax1.imshow(
        pred_img.squeeze(),
        cmap=ListedColormap(["white", "black"]),
        vmin=0,
        vmax=1,
    )
    ax1.set_title("Prediction")
    ax1.axis("off")

    # Target Mask
    if isinstance(target_img, torch.Tensor):
        target_img = target_img.cpu()
    ax2.imshow(
        target_img.squeeze(),
        cmap=ListedColormap(["white", "black"]),
        vmin=0,
        vmax=1,
    )
    ax2.set_title("Target Mask")
    ax2.axis("off")

    # -- Manually add a colorbar axis that does NOT shrink ax0 --
    # Get the position of ax0 in figure coordinates
    pos = ax0.get_position()  # (x0, y0, width, height)

    # place the colobar: just to the right of ax0
    x_cbar = pos.x0 + pos.width + 0.005  # chosen manually
    y_cbar = pos.y0
    cbar_width = 0.01
    cbar_height = pos.height

    # Create a new axis for the colorbar
    cax = fig.add_axes([x_cbar, y_cbar, cbar_width, cbar_height])

    # Draw the colorbar in this new axis
    fig.colorbar(im0, cax=cax)

    plt.show()


def infer_and_visualize(
    model,
    image_tensor,
    mask_tensor,
    apply_sigmoid=True,
    threshold=0.5,
    device=None,
    figsize=(12, 4),
):
    pred_img = infer(
        model=model,
        image_tensor=image_tensor,
        device=device,
        apply_sigmoid=apply_sigmoid,
        threshold=threshold,
    )
    plot_prediction(
        original_img=image_tensor,
        pred_img=pred_img,
        target_img=mask_tensor,
        figsize=figsize,
    )

    return pred_img


def plot_prediction_overlap(
    original_img,
    pred_img,
    target_img,
    figsize=(12, 4),
    cmap="viridis",
    legend=False,
    invert=True,
):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)
    # Original Image
    # At this point this might be a tensor of shape (1, N, N)
    # Squeeze the tensor
    # Also it might be on cuda, for plotting we move to cpu
    # Otherwise, matplotlib will throw an error
    if isinstance(original_img, torch.Tensor):
        original_img = original_img.cpu()
    im0 = ax0.imshow(original_img.squeeze(), cmap=cmap)
    ax0.set_title("Original Image")
    ax0.axis("off")

    if invert:
        pred_img = 1 - pred_img
        target_img = 1 - target_img
    # Prediction
    if isinstance(pred_img, torch.Tensor):
        pred_img = np.array(pred_img.cpu())
    pred_img = pred_img.squeeze()

    masked_pred = np.ma.masked_where(pred_img < 0.5, pred_img)
    ax1.imshow(original_img.squeeze(), cmap=cmap)
    ax1.imshow(
        masked_pred, cmap="Oranges", alpha=1, vmin=0, vmax=2
    )  # Only where pred >= 0.5
    ax1.set_title("Prediction Overlay")
    ax1.axis("off")

    # Target Mask
    if isinstance(target_img, torch.Tensor):
        target_img = np.array(target_img.cpu())
    target_img = target_img.squeeze()
    masked_target = np.ma.masked_where(target_img < 0.5, target_img)
    gray_bg = np.full_like(pred_img, 0.5)  # 0.5 in [0,1] => mid-gray

    ax2.imshow(gray_bg, cmap="gray", vmin=-7, vmax=1)
    ax2.imshow(masked_target, cmap="Blues", alpha=0.6, vmin=-5, vmax=1)
    ax2.imshow(masked_pred, cmap="Oranges", alpha=0.6, vmin=-5, vmax=3)
    ax2.set_title("Prediction & Target Overlay")
    ax2.axis("off")

    if legend:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="orange", edgecolor="orange", label="Prediction"),
            Patch(facecolor="blue", edgecolor="blue", label="Target"),
        ]
        ax2.legend(handles=legend_elements, loc="best")

    # -- Manually add a colorbar axis that does NOT shrink ax0 --
    # Get the position of ax0 in figure coordinates
    pos = ax0.get_position()  # (x0, y0, width, height)

    # place the colobar: just to the right of ax0
    x_cbar = pos.x0 + pos.width + 0.005  # chosen manually
    y_cbar = pos.y0
    cbar_width = 0.01
    cbar_height = pos.height

    # Create a new axis for the colorbar
    cax = fig.add_axes([x_cbar, y_cbar, cbar_width, cbar_height])

    # Draw the colorbar in this new axis
    fig.colorbar(im0, cax=cax)

    plt.show()


def infer_and_plot_overlap(
    model,
    image_tensor,
    mask_tensor,
    apply_sigmoid=True,
    threshold=0.5,
    device=None,
    figsize=(12, 4),
    cmap="viridis",
    legend=False,
    invert=True,
):
    pred_img = infer(
        model=model,
        image_tensor=image_tensor,
        device=device,
        apply_sigmoid=apply_sigmoid,
        threshold=threshold,
    )
    plot_prediction_overlap(
        original_img=image_tensor,
        pred_img=pred_img,
        target_img=mask_tensor,
        figsize=figsize,
        cmap=cmap,
        legend=legend,
        invert=invert,
    )

    return pred_img


def plot_selected_pixels(original_img, pred_img, figsize=(6, 6), invert=True):
    if isinstance(original_img, torch.Tensor):
        original_img = original_img.cpu()

    if isinstance(pred_img, torch.Tensor):
        pred_img = np.array(pred_img.cpu())

    if invert:
        pred_img = 1 - pred_img

    original_img = original_img.squeeze()
    pred_img = pred_img.squeeze()

    # Threshold 0.5 to find pixels selected as target
    masked_pred = np.ma.masked_where(pred_img > 0.5, pred_img)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(original_img, cmap="viridis")
    ax.imshow(
        masked_pred, cmap="Grays", alpha=1, vmin=-1, vmax=2
    )  # Only where pred >= 0.5
    ax.axis("off")
    plt.show()


def plot_img_and_target(img, target, figsize=(12, 6), cmap="viridis"):
    if img.shape[0] == 1:
        img = img.squeeze()

    if target.shape[0] == 1:
        target = target.squeeze()

    fig, axes = plt.subplots(1, 2, figsize=figsize)  # 1 row, 2 columns

    # Plot the first image
    im = axes[0].imshow(img, cmap=cmap)
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


def plot_img_and_target_overlay(
    original_img,
    target_img,
    figsize=(12, 4),
    cmap="viridis",
    invert=True,
):
    """
    Plots the original image and an overlay of the target mask.

    Parameters:
    - original_img: The original image (tensor or numpy array) of shape (1, N, N).
    - target_img: The target mask (tensor or numpy array) of shape (1, N, N).
    - figsize: Figure size.
    - cmap: Colormap for the original image.
    - invert: If True, the target mask is inverted (1 - target) before overlay.
    """
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)

    # Process the original image: if it's a tensor, move to CPU and squeeze
    if isinstance(original_img, torch.Tensor):
        original_img = original_img.cpu()
    orig = original_img.squeeze()
    im0 = ax0.imshow(orig, cmap=cmap)
    ax0.set_title("Original Image")
    ax0.axis("off")

    # Process the target image: if it's a tensor, move to CPU and convert to numpy, then squeeze
    if isinstance(target_img, torch.Tensor):
        target_img = np.array(target_img.cpu())
    target = target_img.squeeze()
    if invert:
        target = 1 - target

    # Mask the target so that only values >= 0.5 are shown
    masked_target = np.ma.masked_where(target < 0.5, target)

    # Plot the original image on the second axis and overlay the masked target in "Blues"
    ax1.imshow(orig, cmap=cmap)
    ax1.imshow(masked_target, cmap="coolwarm", alpha=1, vmin=0, vmax=1)
    ax1.set_title("Target Overlay")
    ax1.axis("off")

    # Optionally, add a colorbar for the original image from ax0.
    pos = ax0.get_position()  # (x0, y0, width, height)
    x_cbar = pos.x0 + pos.width + 0.005  # adjust position as needed
    y_cbar = pos.y0
    cbar_width = 0.01
    cbar_height = pos.height

    cax = fig.add_axes([x_cbar, y_cbar, cbar_width, cbar_height])
    fig.colorbar(im0, cax=cax)

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

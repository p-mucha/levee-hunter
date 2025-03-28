from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, PowerNorm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Union
import warnings


from levee_hunter.modeling.inference import infer


def plot_training_validation_loss(
    train_loss_list: List[float], val_loss_list: List[float], figsize: tuple = (8, 5)
) -> None:
    # Plot training and validation loss
    epochs_range = range(1, len(train_loss_list) + 1)

    plt.figure(figsize=figsize)
    plt.plot(epochs_range, train_loss_list, label="Train Loss")
    plt.plot(epochs_range, val_loss_list, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


# -------------------------- Helper Functions to the Main plot() Function Below -------------------------- #


def plot_original_img(
    ax: Axes,
    image: Union[np.ndarray, torch.Tensor],
    cmap: str = "viridis",
    powernorm_threshold: float = None,
):
    img_scale = image.max().item() - image.min().item()
    if powernorm_threshold is not None and img_scale > powernorm_threshold:
        norm = PowerNorm(gamma=0.5, vmin=image.min().item(), vmax=image.max().item())
        im = ax.imshow(image, norm=norm, cmap=cmap)
        ax.set_title("Original Image (PowerNorm)")
    else:
        im = ax.imshow(image, cmap=cmap)
        ax.set_title("Original Image")
    ax.axis("off")
    return im


def plot_mask(ax: Axes, mask: Union[np.ndarray, torch.Tensor]) -> None:
    if mask is None:
        ax.text(0.5, 0.5, "Mask not provided", fontsize=12, ha="center", va="center")
    else:
        ax.imshow(mask, cmap=ListedColormap(["black", "white"]), vmin=0, vmax=1)
        ax.set_title("Mask")
    ax.axis("off")


def plot_pred(ax: Axes, pred: Union[np.ndarray, torch.Tensor]) -> None:
    if pred is None:
        ax.text(
            0.5, 0.5, "Prediction not provided", fontsize=12, ha="center", va="center"
        )
    else:
        ax.imshow(pred, cmap=ListedColormap(["black", "white"]), vmin=0, vmax=1)
        ax.set_title("Prediction")
    ax.axis("off")


def plot_image_pred_overlay(ax, image, pred, cmap="viridis") -> None:
    if pred is None:
        ax.text(
            0.5, 0.5, "Prediction not provided", fontsize=12, ha="center", va="center"
        )
    else:
        masked_pred = np.ma.masked_where(pred < 0.5, pred)
        ax.imshow(image, cmap=cmap)
        ax.imshow(masked_pred, cmap="Oranges", alpha=1, vmin=0, vmax=2)
        ax.set_title("Image with Prediction Overlay")
    ax.axis("off")


def plot_image_mask_overlay(ax, image, mask, cmap="viridis"):
    if mask is None:
        ax.text(0.5, 0.5, "Mask not provided", fontsize=12, ha="center", va="center")
    else:
        masked_mask = np.ma.masked_where(mask < 0.5, mask)
        ax.imshow(image, cmap=cmap)
        ax.imshow(masked_mask, cmap="coolwarm", alpha=1, vmin=0, vmax=1)
        ax.set_title("Image with Mask Overlay")
    ax.axis("off")


def plot_pred_mask_overlay(ax, image, mask, pred, cmap="viridis"):
    if mask is None:
        ax.text(0.5, 0.5, "Mask not provided", fontsize=12, ha="center", va="center")
    else:
        masked_pred = np.ma.masked_where(pred < 0.5, pred)
        masked_mask = np.ma.masked_where(mask < 0.5, mask)
        gray_bg = np.full_like(image, 0.75)
        ax.imshow(gray_bg, cmap="gray", vmin=0, vmax=1)
        ax.imshow(masked_mask, cmap="coolwarm", alpha=0.6, vmin=0, vmax=1)
        ax.imshow(masked_pred, cmap="Oranges", alpha=0.6, vmin=0, vmax=2)
        ax.set_title("Prediction & Mask Overlay")
    ax.axis("off")


def plot_predicted_pixels(ax, image, pred, cmap="viridis"):
    # Threshold 0.5 to find pixels selected as target
    masked_pred = np.ma.masked_where(pred > 0.5, pred)

    ax.imshow(image, cmap=cmap)
    ax.imshow(
        masked_pred, cmap="Grays", alpha=1, vmin=-1, vmax=2
    )  # Only where pred >= 0.5
    ax.set_title("Image's Pixels Classified as Target")
    ax.axis("off")


def plot(
    image: Union[np.ndarray, torch.Tensor] = None,
    mask: Union[np.ndarray, torch.Tensor] = None,
    pred: Union[np.ndarray, torch.Tensor] = None,
    plot_types: List[str] = None,
    figsize: tuple = (15, 5),
    cmap: str = "viridis",
    inverted: bool = True,
    powernorm_threshold: float = None,
):
    """
    Accepts np.ndarray or torch.Tensor for image, mask, and pred. Performs squeezing, so will accept shapes like:
    (1, H, W), or (1, 1, H, W) or (H, W).

    Inputs need to match plot_types, eg if user selects "image" and "mask", then image and mask cannot be None.

    Inputs:
    - image: the original image. np.ndarray or torch.Tensor, at shape (H, W) or (1, H, W) or (1, 1, H, W)
    - mask: the ground truth mask. np.ndarray or torch.Tensor, at shape (H, W) or (1, H, W) or (1, 1, H, W)
    - pred: the predicted mask. np.ndarray or torch.Tensor, at shape (H, W) or (1, H, W) or (1, 1, H, W)
    - plot_types: list of strings, each string is a type of plot to show. Options are:
        - "image": original image
        - "mask": ground truth mask
        - "pred": predicted mask
        - "image_pred_overlay": original image with prediction overlay
        - "image_mask_overlay": original image with mask overlay
        - "pred_mask_overlay": predicted mask with ground truth mask overlay
        - "predicted_pixels": pixels classified as target in the original image
    - figsize: size of the figure
    - cmap: colormap to use for the original image
    - inverted: whether to invert the mask and prediction (1 - mask/pred), select True if target pixels are 0s and background are 1s.
    - powernorm_threshold: if the image's scale is greater than this, use PowerNorm for the original image

    Outputs:
    - None, but shows the plots in a matplotlib figure.
    """

    if plot_types is None:
        plot_types = ["image"]

    def process_image(img: Union[np.ndarray, torch.Tensor]):
        if img is None:
            return None
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        return img.squeeze()

    # Process the images once so they're ready to use
    image = process_image(image)
    mask = process_image(mask)
    pred = process_image(pred)

    if inverted:
        if mask is not None:
            mask = 1 - mask
        if pred is not None:
            pred = 1 - pred

    n = len(plot_types)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    # Mapping plot type names to functions using lambda to pass the processed images.
    PLOT_FUNCTIONS = {
        "image": lambda ax: plot_original_img(ax, image, cmap, powernorm_threshold),
        "mask": lambda ax: plot_mask(ax, mask),
        "pred": lambda ax: plot_pred(ax, pred),
        "image_pred_overlay": lambda ax: plot_image_pred_overlay(ax, image, pred, cmap),
        "image_mask_overlay": lambda ax: plot_image_mask_overlay(ax, image, mask, cmap),
        "pred_mask_overlay": lambda ax: plot_pred_mask_overlay(
            ax, image, mask, pred, cmap
        ),
        "predicted_pixels": lambda ax: plot_predicted_pixels(ax, image, pred, cmap),
    }

    im0 = None
    for ax, plot_type in zip(axes, plot_types):
        func = PLOT_FUNCTIONS.get(plot_type)
        if func and plot_type == "image":
            im0 = func(ax)

        elif func:
            func(ax)

        else:
            ax.text(
                0.5,
                0.5,
                f"Plot type '{plot_type}' not recognized",
                ha="center",
                va="center",
            )
            ax.axis("off")

    if "image" in plot_types:
        # Get the position of ax0 in figure coordinates
        pos = axes[0].get_position()  # (x0, y0, width, height)

        # Place the colorbar: just to the right of axes[0]
        x_cbar = pos.x0 + pos.width + 0.005  # chosen manually
        y_cbar = pos.y0
        cbar_width = 0.01
        cbar_height = pos.height

        # Create a new axis for the colorbar
        cax = fig.add_axes([x_cbar, y_cbar, cbar_width, cbar_height])

        # Draw the colorbar in this new axis
        fig.colorbar(im0, cax=cax)

    plt.show()


# -------------------------- Deprecated Functions Below -------------------------- #


def plot_prediction(original_img, pred_img, target_img, figsize=(12, 4)):
    warnings.warn(
        "This function is deprecated. Use the plot() with plot_types: image, pred, mask instead."
    )
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
    image,
    mask_tensor,
    apply_sigmoid=True,
    threshold=0.5,
    device=None,
    figsize=(12, 4),
):
    pred_img = infer(
        model=model,
        image=image,
        device=device,
        apply_sigmoid=apply_sigmoid,
        threshold=threshold,
    )
    plot_prediction(
        original_img=image,
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
    warnings.warn(
        "This function is deprecated. Use the plot() with plot_types: image, image_pred_overlay, pred_mask_overlay instead."
    )
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
    warnings.warn(
        "This function is deprecated. Use the plot() with plot_types: predicted_pixels"
    )
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
    warnings.warn(
        "This function is deprecated. Use the plot() with plot_types: image, mask instead."
    )
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
    original_img: np.ndarray | torch.Tensor,
    target_img: np.ndarray | torch.Tensor,
    figsize: tuple = (12, 4),
    cmap: str = "viridis",
    invert: bool = True,
    powernorm_threshold: float = None,
) -> None:
    """
    Plots the original image and an overlay of the target mask.

    Parameters:
    - original_img: The original image (tensor or numpy array) of shape (1, N, N).
    - target_img: The target mask (tensor or numpy array) of shape (1, N, N).
    - figsize: Figure size.
    - cmap: Colormap for the original image.
    - invert: If True, the target mask is inverted (1 - target) before overlay.
    - powernorm_threshold: If not None, the image will be powerscaled if the range of values is higher than the threshold.
    """
    warnings.warn(
        "This function is deprecated. Use the plot() with plot_types: image, image_mask_overlay instead."
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)

    # Process the original image: if it's a tensor, move to CPU and squeeze
    if isinstance(original_img, torch.Tensor):
        original_img = original_img.cpu()
    orig = original_img.squeeze()

    # Powerscale allows to change the scale if range is high
    # user can set powernorm_threshold, if range of values on
    # the image is higher than the threshold, then the image
    # will be powerscaled for better visibility. Only works if not None
    if powernorm_threshold is not None:
        norm = PowerNorm(gamma=0.5, vmin=orig.min().item(), vmax=orig.max().item())
        if orig.max().item() - orig.min().item() > powernorm_threshold:
            im0 = ax0.imshow(orig, norm=norm, cmap=cmap)
            ax0.set_title("Original Image (PowerNorm)")
        else:
            im0 = ax0.imshow(orig, cmap=cmap)
            ax0.set_title("Original Image")
    else:
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


def plot_overlayed_img_mask_pred(
    image,
    mask,
    pred,
    figsize: tuple = (12, 6),
    cmap: str = "viridis",
    invert: bool = True,
):

    warnings.warn(
        "This function is deprecated. Use the plot() with plot_types: image, image_mask_overlay, image_pred_overlay instead."
    )
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=figsize)

    im0 = ax0.imshow(image.squeeze(), cmap=cmap)
    ax0.set_title("Original Image")
    ax0.axis("off")

    if invert:
        pred = 1 - pred
        mask = 1 - mask

    masked_mask = np.ma.masked_where(mask < 0.5, mask)
    ax1.imshow(image.squeeze(), cmap=cmap)
    ax1.imshow(
        masked_mask, cmap="coolwarm", alpha=1, vmin=0, vmax=1
    )  # Only where pred >= 0.5
    ax1.set_title("Mask Overlay")
    ax1.axis("off")

    pred = pred.squeeze()
    masked_pred = np.ma.masked_where(pred < 0.5, pred)
    ax2.imshow(image.squeeze(), cmap=cmap)
    ax2.imshow(
        masked_pred, cmap="Oranges", alpha=1, vmin=0, vmax=2
    )  # Only where pred >= 0.5
    ax2.set_title("Prediction Overlay")
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

import numpy as np
import torch
from torch import nn
from typing import Union, Optional


def infer(
    model: nn.Module,
    image: Union[np.ndarray, torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
    apply_sigmoid: bool = True,
    threshold: float = 0.5,
):
    """
    Run inference on the given image or images using the model and device.
    Recommended input shape: (N, C, H, W).
    Returns the model's output as a numpy array on the CPU, shape (N, C, H, W).

    Inputs:
    - model: The PyTorch model to use for inference. We use segmentation_models.pytorch.
    - image: The input image to run inference on. Can be np.array or torch.Tensor, recommended shape is (N, C, H, W).
    - device: The device to run the model on. If None, it will use 'cuda' if available, otherwise 'cpu'.
    - apply_sigmoid: Whether to apply sigmoid activation to the model's output.
    - threshold: The threshold to use for binarizing the output. Default is 0.5.

    Outputs:
    - output: The model's output as a numpy array on the CPU, always will have shape (N, C, H, W).
        eg For simgle image and one channel: (1, 1, H, W).
    """

    # Ensure we are in evaluation mode
    model.eval()

    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)

    # If the tensor is 2D, assume it's a grayscale image and add a channel dimension
    if len(image.shape) == 2:
        image = image.unsqueeze(0)

    # If the tensor is 3D, assume it's (C, H, W) for a single image, so add a batch dimension
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    # Determine device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move the image to the device
    image = image.to(device)

    with torch.no_grad():
        # Now image is now a torch.Tensor with shape (N, C, H, W)
        output = model(image)
        if apply_sigmoid:
            output = torch.sigmoid(output)
            output = (output > threshold).float()

    # Move the output back to CPU
    output = output.cpu().numpy()

    return output


def get_preds_targets(model, val_loader, device, invert=True):
    all_preds = []
    all_targets = []
    threshold = 0.5

    model.eval()
    with torch.no_grad():
        for images, masks, *_ in val_loader:
            # Shapes are (batch_size, 1, height, width) for both
            # Note for some loaders batch_size is len(images)
            preds = infer(
                model,
                image=images,
                device=device,
                apply_sigmoid=True,
                threshold=threshold,
            )

            if len(preds.shape) == 3:
                preds = preds.reshape(1, *preds.shape)

            # Store results for evaluation
            all_preds.append(preds)
            all_targets.append(masks)
            # print(all_preds[0].shape, all_targets[0].shape)

    # Concatenate all batches into single tensors
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    if invert:
        # Invert target masks if needed (ensure foreground is 1, background is 0)
        all_targets = torch.tensor(1 - all_targets)
        all_preds = torch.tensor(1 - all_preds)

    # Convert to integer type (0 or 1) to satisfy `get_stats` requirements
    all_preds = all_preds.long()
    all_targets = all_targets.long()

    return all_preds, all_targets

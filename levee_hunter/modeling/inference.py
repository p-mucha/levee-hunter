import numpy as np
import torch


def infer(model, image_tensor, device=None, apply_sigmoid=True, threshold=0.5):
    """
    Run inference on the given image_tensor using the model and device.
    Returns the model's output as a numpy array on the CPU.

    Expected input:
      - Single image: shape (C, H, W)
      - Multiple images: shape (N, C, H, W)

    If a single image is passed, the function adds a batch dimension.
    """
    model.eval()

    # Convert to tensor if necessary
    if not isinstance(image_tensor, torch.Tensor):
        image_tensor = torch.tensor(image_tensor)

    # If the tensor is 2D, assume it's a grayscale image and add a channel dimension
    if len(image_tensor.shape) == 2:
        image_tensor = image_tensor.unsqueeze(0)

    # If the tensor is 3D, assume it's (C, H, W) for a single image, so add a batch dimension
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Determine device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move the image tensor to the device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        # Now image_tensor is shape (N, C, H, W)
        output = model(image_tensor)
        if apply_sigmoid:
            output = torch.sigmoid(output)
            output = (output > threshold).float()

    # Move the output back to CPU
    output = output.cpu().numpy()

    # Optionally, if you want to squeeze out singleton dimensions from a single image
    if output.shape[0] == 1:
        output = output.squeeze(0)

    return output


def get_preds_targets(model, val_loader, device, invert=True):
    all_preds = []
    all_targets = []
    threshold = 0.5

    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            # Shapes are (batch_size, 1, height, width) for both
            # Note for some loaders batch_size is len(images)
            preds = infer(
                model,
                image_tensor=images,
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

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

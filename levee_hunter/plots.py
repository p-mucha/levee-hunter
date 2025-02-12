import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def infer_and_visualize(model, image_tensor, mask_tensor, apply_sigmoid=True):

    if apply_sigmoid:
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))
            output = torch.sigmoid(output).cpu().squeeze().numpy()
            output = (output > 0.5).astype(np.uint8)
    else:
        with torch.no_grad():
            output = model(image_tensor.unsqueeze(0))
            output = np.array(output).reshape(256,256)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image_tensor.reshape(256, 256))

    plt.subplot(1, 3, 2)
    plt.title("Prediction")
    plt.imshow(output, cmap=ListedColormap(["white", "black"]), vmin=0, vmax=1)

    plt.subplot(1, 3, 3)
    plt.title("Target Mask")
    plt.imshow(mask_tensor.reshape(256, 256), cmap=ListedColormap(["white", "black"]), vmin=0, vmax=1)

    plt.show()

    return output
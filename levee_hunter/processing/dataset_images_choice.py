from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import rioxarray
import time
import warnings

from levee_hunter.plots import plot_img_and_target
from levee_hunter.processing.apply_mask_type import apply_mask_type


def interactive_images_selection(
    intermediate_data_path: str,
    output_dir: str,
    dilation_size: int = 10,
    figsize: tuple = (12, 6),
) -> None:
    """
    Allows the user to interactively select images to keep, remove, or mark as special.

    Inputs:
    - intermediate_data_path: str, path to the intermediate directory, should have the images and masks directories.
    - output_dir: str, path to the output directory.
    - dilation_size: int, size of the dilation kernel, Visualisation ONLY.

    Outputs:
    - None, saves the selected images to the output directory.
    """
    # This should be directory with images and masks directories creatd at
    # the initial processing step, where we split the images
    intermediate_data_path = Path(intermediate_data_path)

    # This should be the processed directory where we will move the selected images
    # for example something like /path/to/data/processed/{resolution}_{size}
    output_dir = Path(output_dir)

    if not intermediate_data_path.exists():
        raise ValueError("Intermediate data path does not exist.")

    if not intermediate_data_path.is_dir():
        raise ValueError("Intermediate data path should be a directory.")

    if not "processed" in output_dir.parts:
        warnings.warn(
            "The output directory is not inside the /data/processed directory. This is not recommended."
        )

    # images dir holds .tif files
    # masks dir holds .npy files
    images_dir = intermediate_data_path / "images"
    masks_dir = intermediate_data_path / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        raise ValueError(
            "Images and targets directories should exist within the intermediate_data_path."
        )

    # This will return a list of full paths to the files
    tif_files = list(images_dir.glob("*.tif"))
    mask_files = list(masks_dir.glob("*.npy"))

    # if no files found, error
    if len(tif_files) == 0 or len(mask_files) == 0:
        raise ValueError("No TIF files found in images or targets directories.")

    print(f"Found {len(tif_files)} images and {len(mask_files)} masks.")
    print("\n ---------------Starting interactive images selection.--------------- \n")
    # Wait for 2 seconds before proceeding.
    time.sleep(2)

    # Loop over tif files inside the images directory
    for i in range(len(tif_files)):
        # Clear previous output while keeping logs
        clear_output(wait=True)

        # For current tif file, get the corresponding mask file
        # using the characteristic filename which is the same for both
        # this filename is N_fileID
        current_tif_file = tif_files[i]
        current_filename = current_tif_file.stem
        current_mask_file = masks_dir / f"{current_filename}.npy"

        # Load the image and mask
        current_img = rioxarray.open_rasterio(str(current_tif_file))
        current_mask = np.load(current_mask_file)

        # Need to dilate before plotting, this is only for visualization
        current_mask = apply_mask_type(
            mask=current_mask,
            mask_type="dilated",
            dilation_size=dilation_size,
            inverted=True,
        )
        print(f"Progress: {i+1}/{len(tif_files)} \n")
        print(f"Currently Processing: {current_tif_file}")
        # plot them side by side
        plot_img_and_target(
            current_img.values.squeeze(), current_mask.squeeze(), figsize=figsize
        )

        # Add an extra plt.pause to ensure previous figures are removed
        plt.pause(0.15)  # Small pause to let Jupyter process figure update

        # Ask user
        user_input = (
            input(
                f"Part {i+1}/{len(tif_files)} - [keep / special / remove / quit] -> (a / w / d / q)? "
            )
            .strip()
            .lower()
        )

        if user_input == "q" or user_input == "quit":
            return

        elif user_input == "keep" or user_input == "a":
            print("Moved to processed")
            # Ensure the "images" subdirectory exists.
            output_images_dir = output_dir / "images"
            output_images_dir.mkdir(parents=True, exist_ok=True)
            # Construct the new filename by appending "_w1" before the extension.
            tif_output = (
                output_images_dir
                / f"{current_tif_file.stem}_w1{current_tif_file.suffix}"
            )

            # Ensure the "masks" subdirectory exists.
            output_masks_dir = output_dir / "masks"
            output_masks_dir.mkdir(parents=True, exist_ok=True)
            # Construct the new filename by appending "_w1" before the extension.
            mask_output = (
                output_masks_dir
                / f"{current_mask_file.stem}_w1{current_mask_file.suffix}"
            )

            current_tif_file.rename(tif_output)
            current_mask_file.rename(mask_output)

        elif user_input == "special" or user_input == "w":
            print("Moved to processed, with weight 2")
            # Ensure the "images" subdirectory exists.
            output_images_dir = output_dir / "images"
            output_images_dir.mkdir(parents=True, exist_ok=True)
            # Construct the new filename by appending "_w1" before the extension.
            tif_output = (
                output_images_dir
                / f"{current_tif_file.stem}_w2{current_tif_file.suffix}"
            )

            # Ensure the "masks" subdirectory exists.
            output_masks_dir = output_dir / "masks"
            output_masks_dir.mkdir(parents=True, exist_ok=True)
            # Construct the new filename by appending "_w1" before the extension.
            mask_output = (
                output_masks_dir
                / f"{current_mask_file.stem}_w2{current_mask_file.suffix}"
            )

            # Move the files to processed
            current_tif_file.rename(tif_output)
            current_mask_file.rename(mask_output)

        elif user_input == "remove" or user_input == "d":
            pass  # do nothing, skip this image

        else:
            print("Invalid input. Please try again.")
            continue  # re-prompt the same part

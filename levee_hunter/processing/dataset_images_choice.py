from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import rioxarray

from levee_hunter.plots import plot_img_and_target


def interactive_images_selection(intermediate_data_path: str, output_dir: str) -> None:
    # This should be directory with images and masks directories creatd at
    # the initial processing step, where we split the images
    intermediate_data_path = Path(intermediate_data_path)

    if not intermediate_data_path.exists():
        raise ValueError("Intermediate data path does not exist.")

    if not intermediate_data_path.is_dir():
        raise ValueError("Intermediate data path should be a directory.")

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

        print(f"Currently processing: {current_filename}")

        # Need to dilate before plotting

        # plot them side by side
        plot_img_and_target(current_img.values.squeeze(), current_mask.squeeze())

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

        elif user_input == "special" or user_input == "w":
            # rename to w2
            print("Moved to special")

        elif user_input == "remove" or user_input == "d":
            pass  # do nothing, skip this image

        else:
            print("Invalid input. Please try again.")
            continue  # re-prompt the same part

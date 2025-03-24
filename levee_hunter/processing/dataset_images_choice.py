from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import rioxarray
import time
from typing import Callable, Optional
import warnings

from levee_hunter.plots import plot_img_and_target, plot_img_and_target_overlay
from levee_hunter.processing.apply_mask_type import apply_mask_type
from levee_hunter.processing.processing_utils import filter_single_image_by_overlap


def interactive_images_selection(
    intermediate_data_path: str,
    output_dir: str,
    dilation_size: int = 10,
    figsize: tuple = (12, 6),
    cmap: str = "viridis",
    plot: str = "overlay",
    file_ids_toprocess: list = None,
    powernorm_threshold: float = None,
    store_bad_bounds: bool = False,
    helper: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
) -> None:
    """
    Allows the user to interactively select images to keep, remove, or mark as special.

    Inputs:
    - intermediate_data_path: str, path to the intermediate directory, should have the images and masks directories.
    - output_dir: str, path to the output directory.
    - dilation_size: int, size of the dilation kernel, Visualisation ONLY.
    - figsize: tuple, size of the figure.
    - cmap: str, colormap to use for plotting.
    - plot: str, options are: None, 'overlay', 'side_by_side'.
    - file_ids_toprocess: list, list of file IDs to process. If None, all files will be processed.
    - powernorm_threshold: float, if not None, the image plot will be powerscaled if the range of values is higher than the threshold.
    - store_bad_bounds: bool, if True, will store the bounds of the images that were not selected in a separate file named bad_bounds.txt.
    - helper: a function that takes in the image and mask (1, H, W) and does something with it.
        For print warning if image is in bad_bounds.txt, or use model to help see missing levees.

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
            "Images and masks directories should exist within the intermediate_data_path."
        )

    # This will return a sorted list of full paths to the files
    # although actually sorting is not necessary, because we get
    # mask filename based on the current tif filename anyway
    tif_files = sorted(images_dir.glob("*.tif"))
    mask_files = sorted(masks_dir.glob("*.npy"))

    # if no files found, error
    if len(tif_files) == 0 or len(mask_files) == 0:
        raise ValueError("No TIF files found in images or targets directories.")

    # Allow user to provide a list of file_IDs of files they want to process
    # Choose only those tifs which have file ID in the provided list.
    # This is a simple way for multiple users to be selecting images at the same time.
    # Since we get mask files based on the current tif file name, there is no need to
    # filter mask files in the same way.
    if file_ids_toprocess:
        if not isinstance(file_ids_toprocess, list):
            raise ValueError("file_ids_toprocess should be a list of strings.")
        if not isinstance(file_ids_toprocess[0], str):
            raise ValueError("file_ids_toprocess should be a list of strings.")
        # Filter tif files based on the file_ids_toprocess
        # tif_file.stem.split('_')[:-1] gives the parts before _p{index}.
        # (this assumes there is no weighting added yet, which should be true for intermediate data)
        # so we join these parts and check if they are in the file_ids_toprocess
        tif_files = [
            tif_file
            for tif_file in tif_files
            if "_".join(tif_file.stem.split("_")[:-1]) in file_ids_toprocess
        ]
        print(f"Found {len(tif_files)} images to process.")
    else:
        print(f"Found {len(tif_files)} images and {len(mask_files)} masks.")

    print("\n ---------------Starting interactive images selection.--------------- \n")
    # Wait for 2 seconds before proceeding.
    time.sleep(2)

    def update_bounds_file(bounds_file_path, current_img_5070):
        bounds = (
            current_img_5070.rio.bounds()
        )  # e.g. (-2198313.241384246, 1234567.89, -2198313.241384245, 1234567.90)
        # Create a comma-separated string without parentheses
        bounds_str = ",".join(map(str, bounds))
        with open(bounds_file_path, "a") as file:
            file.write(f"{bounds_str}\n")

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

        #########################################
        # At this step, we check if our current_img overlaps with any previously accepted images
        # If it does, we skip this image and move to the next one
        # This is done to avoid having multiple images of the same area in the dataset
        # First we will always convert to the same CRS, then using the existing list of bounds
        # in the bounds.txt, we check for overlap, if there is no overlap,
        # we present the image to the user, if user accepts it, we need to update
        # the bounds.txt file with the new bounds, and the file is moved to processed
        current_img_5070 = current_img.rio.reproject("EPSG:5070")

        bounds_file_path = output_dir / "bounds.txt"

        # Create an empty bounds file if it does not exist
        if not bounds_file_path.exists():
            # make parent dir if does not exist yet
            bounds_file_path.parent.mkdir(parents=True, exist_ok=True)
            bounds_file_path.touch()

        no_overlap = filter_single_image_by_overlap(
            tif_img=current_img_5070, bounds_file=bounds_file_path, threshold=10
        )

        # if this option is selected, similarly to bounds of good images,
        # we also store bounds of rejected images
        # note: only rejected by the user, if overlap check not passed, then
        # it is not stored.
        if store_bad_bounds:
            bad_bounds_file_path = output_dir / "bad_bounds.txt"
            if not bad_bounds_file_path.exists():
                bad_bounds_file_path.touch()

        # if there is overlap, skip this image, go to next loop iteration
        # (next tig file)
        if no_overlap == False:
            # remove file and go to the next one
            current_tif_file.unlink()
            current_mask_file.unlink()

            # Wait for 1.5 seconds before proceeding.
            time.sleep(1.5)
            continue

        # Need to dilate before plotting, this is only for visualization
        current_mask = apply_mask_type(
            mask=current_mask,
            mask_type="dilated",
            dilation_size=dilation_size,
            inverted=True,
        )
        print(f"Progress: {i+1}/{len(tif_files)} \n")
        print(f"Currently Processing: {current_tif_file}")

        if helper is not None:
            # current_img and current_mask have shapes (1, H, W) currently
            helper(current_img.values, current_mask)

        # -------------------------------- Plotting Here --------------------------------
        if plot is not None:
            if plot not in ["overlay", "side_by_side"]:
                raise ValueError("plot should be either 'overlay' or 'side_by_side'.")
            if plot == "overlay":
                plot_img_and_target_overlay(
                    original_img=current_img.values.squeeze(),
                    target_img=current_mask.squeeze(),
                    figsize=figsize,
                    cmap=cmap,
                    invert=True,
                    powernorm_threshold=powernorm_threshold,
                )
            elif plot == "side_by_side":
                # plot them side by side
                plot_img_and_target(
                    current_img.values.squeeze(),
                    current_mask.squeeze(),
                    figsize=figsize,
                    cmap=cmap,
                )
        else:
            if helper is None:
                warnings.warn(
                    "No plotting function provided. Either set the plot or helper arguments."
                )

        # Add an extra plt.pause to ensure previous figures are removed
        plt.pause(0.15)  # Small pause to let Jupyter process figure update
        # --------------------------------------------------------------------------------

        # Ask user
        user_input = (
            input(
                f"Part {i+1}/{len(tif_files)} - [keep / special / remove / quit] -> (a / w / d / q)? "
            )
            .strip()
            .lower()
        )

        output_images_dir = output_dir / "images"
        output_masks_dir = output_dir / "masks"

        if user_input == "q" or user_input == "quit":
            return

        elif user_input == "keep" or user_input == "a":
            print("Moved to processed")
            # Ensure the "images" subdirectory exists.
            output_images_dir.mkdir(parents=True, exist_ok=True)
            # Construct the new filename by appending "_w1" before the extension.
            tif_output = (
                output_images_dir
                / f"{current_tif_file.stem}_w1{current_tif_file.suffix}"
            )

            # Ensure the "masks" subdirectory exists.
            output_masks_dir.mkdir(parents=True, exist_ok=True)
            # Construct the new filename by appending "_w1" before the extension.
            mask_output = (
                output_masks_dir
                / f"{current_mask_file.stem}_w1{current_mask_file.suffix}"
            )

            current_tif_file.rename(tif_output)
            current_mask_file.rename(mask_output)

            # Update the bounds file with the new bounds
            update_bounds_file(bounds_file_path, current_img_5070)

        elif user_input == "special" or user_input == "w":
            print("Moved to processed, with weight 2")
            # Ensure the "images" subdirectory exists.
            output_images_dir.mkdir(parents=True, exist_ok=True)
            # Construct the new filename by appending "_w1" before the extension.
            tif_output = (
                output_images_dir
                / f"{current_tif_file.stem}_w2{current_tif_file.suffix}"
            )

            # Ensure the "masks" subdirectory exists.
            output_masks_dir.mkdir(parents=True, exist_ok=True)
            # Construct the new filename by appending "_w1" before the extension.
            mask_output = (
                output_masks_dir
                / f"{current_mask_file.stem}_w2{current_mask_file.suffix}"
            )

            # Move the files to processed
            current_tif_file.rename(tif_output)
            current_mask_file.rename(mask_output)

            # Update the bounds file with the new bounds
            update_bounds_file(bounds_file_path, current_img_5070)

        elif user_input == "remove" or user_input == "d":
            if store_bad_bounds:
                update_bounds_file(bad_bounds_file_path, current_img_5070)

            # remove file and go to the next one
            current_tif_file.unlink()
            current_mask_file.unlink()
            pass  # do nothing, skip this image, do not update bounds

        else:
            print("Invalid input. Please try again.")
            continue  # re-prompt the same part

import argparse
import geopandas as gpd
import numpy as np
import os
import sys
from pathlib import Path
import rioxarray
from tqdm import tqdm

from levee_hunter.database_management import generate_file_id_from_name
from levee_hunter.get_mask import get_mask
from levee_hunter.paths import find_project_root
from levee_hunter.processing.processing_utils import (
    split_images,
    remove_invalid_images,
    remove_empty_images,
)
from levee_hunter.utils import get_processing_config


def main():
    # load the configuration
    parser = argparse.ArgumentParser(description="Processing pipeline configuration")
    parser.add_argument(
        "--config_name",
        type=str,
        default="default",
        help="Name of the processing configuration",
    )
    args = parser.parse_args()
    config = get_processing_config(args.config_name)
    print(
        "\n -------------------- Beginning processing pipeline -------------------- \n"
    )
    print("Loaded configuration:", config)

    resolution = config["resolution"]
    final_size = config["final_size"]

    # find the project's root directory
    root_dir = config["root_dir"]
    if root_dir is None:
        root_dir = find_project_root()
        print("\n Root directory not specified. Using project root:", root_dir)
    else:
        root_dir = Path(root_dir)

    # find the raw data directory
    input_dir = root_dir / config["input_dir"]
    tif_files = [file for file in os.listdir(input_dir) if file.endswith(".tif")]
    if len(tif_files) == 0:
        print(f"\n No TIF files found in directory: {input_dir}. Exiting.")
        sys.exit(0)
    print(f"\n Found {len(tif_files)} tif files. \n")

    # load levees data
    levees_path = config["levees_path"]
    if levees_path is None:
        levees_path = root_dir / "data/raw/levees/levees.gpkg"

    levees_path = Path(levees_path)
    # check if levees file exists
    if not levees_path.exists():
        raise FileNotFoundError(f"Levees file not found: {levees_path}")

    levees_data = gpd.read_file(levees_path)

    # loop over the tif files
    print("\n")
    for current_tif_file in tqdm(tif_files, desc="Processing TIF files"):

        # load current tif file as rioxarray
        current_img = rioxarray.open_rasterio(str(input_dir / current_tif_file))

        # get mask for current tif, remember mask has shape (H, W)
        # Note, levees do not have to be in the same crs as tif_image, the
        # get_mask function will take care of this
        _, mask = get_mask(
            tif_image=current_img,
            levees=levees_data,
            invert=config["invert"],
            mask_type=config["mask_type"],
            dilation_size=config["dilation_size"],
            gaussian_sigma=config["gaussian_sigma"],
        )

        # split large images and their masks into smaller patches
        # Note images should be a list, but split_images will automatically take care of this if single
        # image is provided.
        # The outputs is a tuple of two lists. First list holds the smaller images in the form of xarray.DataArray
        # and the second list holds the smaller masks in the form of numpy arrays.
        # smaller image and smaller target, both have shapes (1, H, W).
        smaller_images, smaller_targets = split_images(
            images=current_img,
            targets=mask,
            final_size=config["final_size"],
            overlap=config["overlap"],
        )

        # If no valid patches remain after filtering, skip this tif file.
        if not smaller_images:
            continue

        # remove invalid images and their corresponding masks
        smaller_images, smaller_targets = remove_invalid_images(
            images=smaller_images, targets=smaller_targets
        )

        # If no valid patches remain after filtering, skip this tif file.
        if not smaller_images:
            continue

        # remove empty images and their corresponding masks, keep some fraction of empty images
        # determined by keep_empty parameter
        smaller_images, smaller_targets = remove_empty_images(
            images=smaller_images,
            targets=smaller_targets,
            keep_empty=config["keep_empty"],
            inverted=config["invert"],
        )

        # If no valid patches remain after filtering, skip this tif file.
        if not smaller_images:
            continue

        # ------------------ Saving the images and masks ------------------#

        # save the smaller images and masks
        # First for current file we get an unique ID.
        # Then every smaller image will be named based on its index and current ID
        # for example 0th image from the list of smaller images will be named as 0_current_ID.tif
        # Where current_ID will be something like: 'c067e80f74'
        # The masks will not store geo information, so we can save them as numpy arrays
        # so they will have the same name but different format: 0_current_ID.npy
        current_file_id = generate_file_id_from_name(current_tif_file)

        # if output_dir is specified, then save the images and masks there
        if config["output_dir"] is not None:
            images_dir = (
                root_dir
                / Path(config["output_dir"])
                / f"{resolution}_{final_size}/images"
            )
            masks_dir = (
                root_dir
                / Path(config["output_dir"])
                / f"{resolution}_{final_size}/masks"
            )
        else:
            # Default option if None
            # Inside intermediate, there will be directory for dataset at given {resolution}_{final_size}
            # Inside this directory, there will be images and masks directory
            images_dir = (
                root_dir / f"data/intermediate/{resolution}_{final_size}/images"
            )
            masks_dir = root_dir / f"data/intermediate/{resolution}_{final_size}/masks"

        # Create directories if they do not exist
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Save the images and masks
        for i in range(len(smaller_images)):
            current_image = smaller_images[i]
            current_mask = smaller_targets[i]

            current_image.rio.to_raster(images_dir / f"{i}_{current_file_id}.tif")

            np.save(masks_dir / f"{i}_{current_file_id}.npy", current_mask)

    print(f"\n Processing Finished. \n Saved images and masks at {images_dir.parent}")


if __name__ == "__main__":
    main()

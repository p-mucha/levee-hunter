import atexit
import geopandas as gpd
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import numpy as np
import os
import rioxarray
import sqlite3
import sys
import time
import torch

from levee_hunter.database_management import get_files_by_state, move_file_state
from levee_hunter.get_mask import get_mask
from levee_hunter.segmentation_dataset import SegmentationDataset
from levee_hunter.augmentations import TRANSFORMS
from levee_hunter.paths import find_project_root, check_if_file_exists


def load_and_split(current_file_name, levees_data, data_dir, size, overlap):
    current_file_path = os.path.join(data_dir, current_file_name)

    img = rioxarray.open_rasterio(current_file_path)

    # mask_type set to None for now, so that the database is general and user can change it
    # later as they wish, to dilated or goussian etc
    lidar_data, target = get_mask(
        img, levees_data.to_crs(img.rio.crs), invert=True, mask_type=None
    )

    if len(target.shape) == 2:
        target = target.reshape(1, *target.shape)

    img_dataset = SegmentationDataset(
        images=lidar_data,
        targets=target,
        transform=None,
        split=True,
        patch_size=size,
        final_size=size,
        overlap=overlap,
    )

    img_dataset.remove_invalid_images()
    img_dataset.remove_empty(keep_empty=0.2)

    return img_dataset


def interactive_dataset_creation(
    db_path,
    levees_file_path,
    resolution="1m",
    size=1056,
    overlap=26,
    dilation_s=10,
    figsize=(6, 6),
    cmap="viridis",
):
    """
    A text-based interactive workflow for splitting images into train_test,
    validation, or discarding (bad) images. Prompts the user via 'input()'
    rather than GUI buttons, so it can run in a console (e.g. on an HPC cluster).
    """

    # Safety check for resolution
    if resolution not in ["1m", "13"]:
        raise ValueError("Resolution must be either '1m' or '13'.")

    # Prepare paths
    tifs_path = find_project_root() / f"data/raw/{resolution}_resolution"

    # Database lock
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA locking_mode = EXCLUSIVE;")
    conn.commit()

    # Ensure database is unlocked when the script exits (even on errors)
    def release_db_lock():
        print("Releasing database lock (using release_db_lock)...")
        conn.close()

    # Register cleanup function to always run on exit
    atexit.register(release_db_lock)

    # Some state variables
    temp_selected_parts = []
    special_selected_parts = []
    file_index = 0
    part_index = 0
    current_file_id = None
    selected_list = None

    # Datasets
    parts_dataset = None
    good_dataset = None
    bad_dataset = None

    # Load levees data
    levees_data = gpd.read_file(levees_file_path)

    # -----------------------------------------------------
    # Helper functions
    def finalize_output():
        print("Process finished.")

    def revert_file_to_unused():
        """If a file is partially processed, revert it back to 'unused'."""
        nonlocal current_file_id
        if selected_list is not None and current_file_id is not None:
            print(f"Reverting file {current_file_id} back to 'unused'...")
            move_file_state(conn, current_file_id, "unused")

    def next_file():
        """Move to the next file or finish the process."""

        clear_output(wait=True)  # Clear output when moving to next file

        nonlocal file_index, part_index, selected_list, current_file_id, parts_dataset
        file_index += 1
        part_index = 0
        selected_list = None
        current_file_id = None
        parts_dataset = None

        if file_index < len(unused_files):
            process_file()
        else:
            finalize_output()

    def finalize_file_selection():
        """Create 'good' and 'bad' datasets, then save them."""
        nonlocal selected_list, temp_selected_parts, special_selected_parts
        nonlocal good_dataset, bad_dataset, parts_dataset

        selected_set = set(temp_selected_parts)
        special_indices_set = set(special_selected_parts)
        all_indices = set(range(len(parts_dataset)))
        bad_indices = list(all_indices - selected_set)

        good_imgs = np.array([parts_dataset.images[i] for i in selected_set])
        good_targets = np.array([parts_dataset.targets[i] for i in selected_set])
        bad_imgs = np.array([parts_dataset.images[i] for i in bad_indices])
        bad_targets = np.array([parts_dataset.targets[i] for i in bad_indices])

        # Adjust weighting for "special" patches
        weights = torch.ones(len(good_imgs))
        for i, idx in enumerate(selected_set):
            if idx in special_indices_set:
                weights[i] = 2.0

        current_file_ids = [current_file_id] * len(good_imgs)

        good_dataset = SegmentationDataset(
            images=good_imgs,
            targets=good_targets,
            transform="train_transform",
            split=False,
            weights=weights,
            file_ids=current_file_ids,
        )
        bad_dataset = SegmentationDataset(
            images=bad_imgs,
            targets=bad_targets,
            transform="normalize_only",
            split=False,
            weights=None,
            file_ids=None,
        )
        good_dataset.overlap = overlap
        bad_dataset.overlap = overlap

        # Save
        dir_path = find_project_root() / f"data/intermediate/{resolution}_{size}"
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save good dataset to train_test or validation
        # Only save if length > 0, otherwise we would concatenate 1 dim with 3 or 4 dims
        if selected_list in ["train_test", "validation"] and len(good_dataset) > 0:
            out_file = f"{selected_list}.pth"  # "train_test.pth" or "validation.pth"
            path_to_good = dir_path / out_file

            # If dataset already exists, update it
            if check_if_file_exists(path_to_good):
                old_dataset = torch.load(path_to_good, weights_only=False)
                if old_dataset.overlap != overlap:
                    raise ValueError("Overlap mismatch with existing dataset.")
                good_dataset = old_dataset + good_dataset

            torch.save(good_dataset, path_to_good)

        # Save bad dataset if non-empty
        if len(bad_dataset) > 0:
            path_to_bad = dir_path / "bad.pth"
            if check_if_file_exists(path_to_bad):
                old_bad = torch.load(path_to_bad, weights_only=False)
                bad_dataset = old_bad + bad_dataset
            torch.save(bad_dataset, path_to_bad)

        print(f'Saved datasets at "{dir_path}".')
        time.sleep(4)

        # Reset
        temp_selected_parts.clear()
        special_selected_parts.clear()
        good_dataset = None
        bad_dataset = None
        next_file()

    def process_parts():
        """Iterates through each part for the current file, prompting user keep/special/remove/q."""
        nonlocal part_index, parts_dataset

        while part_index < len(parts_dataset):
            # Clear previous output while keeping logs
            clear_output(wait=True)

            # Close previous matplotlib figures BEFORE plotting a new one
            plt.close("all")

            # Print log messages so they persist
            print(f"Processing file {file_index + 1}/{len(unused_files)}")
            print(f"File ID: {current_file_id}")
            print(f"Filename: {unused_files[file_index][1]}")
            print(f"Processing part {part_index+1}/{len(parts_dataset)}")

            # Call `plot()` method (which internally calls `plt.show()`)
            parts_dataset.plot(
                part_index,
                figsize=figsize,
                mask_type="dilated",
                dilation_size=dilation_s,
                cmap=cmap,
            )

            # Add an extra plt.pause to ensure previous figures are removed
            plt.pause(0.05)  # Small pause to let Jupyter process figure update

            # Ask user
            user_input = (
                input(
                    f"Part {part_index+1}/{len(parts_dataset)} - [keep / special / remove / quit] -> (a / w / d / q)? "
                )
                .strip()
                .lower()
            )

            if user_input == "q" or user_input == "quit":
                # Quit mid-file and revert it
                revert_file_to_unused()
                close_connection()
                return

            elif user_input == "keep" or user_input == "a":
                temp_selected_parts.append(part_index)

            elif user_input == "special" or user_input == "w":
                temp_selected_parts.append(part_index)
                special_selected_parts.append(part_index)

            elif user_input == "remove" or user_input == "d":
                pass  # do nothing, skip

            else:
                print("Invalid input. Please try again.")
                continue  # re-prompt the same part

            part_index += 1

        # If we finished all parts
        if part_index >= len(parts_dataset):
            print("Finalizing file selection...")
            finalize_file_selection()

    def process_file():
        """Handle the current file: ask user train_test/validation/q, then split if needed."""
        nonlocal file_index, current_file_id, selected_list, parts_dataset, part_index

        # If we've exhausted files, we can end
        if file_index >= len(unused_files):
            print("No more unused files.")
            close_connection()
            return

        file_id, filename = unused_files[file_index]
        print(f"Processing file {file_index + 1}/{len(unused_files)}")
        print(f"File ID: {file_id}")
        print(f"Filename: {filename}")

        # Prompt user
        while True:
            user_input = (
                input("Choose [train_test / validation / quit] (t / v / q): ")
                .strip()
                .lower()
            )

            if user_input == "q" or user_input == "quit":
                revert_file_to_unused()
                close_connection()
                return

            elif user_input in ["t", "v", "train_test", "validation"]:
                if user_input == "t":
                    user_input = "train_test"
                elif user_input == "v":
                    user_input = "validation"

                selected_list = user_input
                current_file_id = file_id
                move_file_state(conn, current_file_id, user_input)

                # Load & split
                print("Splitting file, please wait...")
                parts_dataset = load_and_split(
                    filename,
                    levees_data,
                    data_dir=tifs_path,
                    size=size,
                    overlap=overlap,
                )
                print(f"Number of parts: {len(parts_dataset)}")

                part_index = 0
                process_parts()
                return  # End file processing

            else:
                print("Invalid input. Please try again.")

    def close_connection():
        """Close db connection and print a message, then exit function."""
        print("Releasing database lock...")
        conn.close()
        sys.exit(
            0
        )  # End the script if desired, or just 'return' if you want partial usage.

    # -----------------------------------------------------
    # Wrap in try-except to handle user interruptions
    try:
        # Fetch unused files
        unused_files = get_files_by_state(conn, "unused")

        if not unused_files:
            print("No 'unused' files found.")
            release_db_lock()
            return

        print("Starting interactive dataset creation...")
        process_file()

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Unlocking database...")
        release_db_lock()
        sys.exit(0)

    except Exception as e:
        print(f"An error occurred: {e}")
        release_db_lock()
        sys.exit(1)

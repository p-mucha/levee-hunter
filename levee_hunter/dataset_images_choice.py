import rioxarray
import sqlite3
import os
import geopandas as gpd
import torch
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
from pathlib import Path

from levee_hunter.database_management import get_files_by_state, move_file_state
from levee_hunter.get_mask import get_mask
from levee_hunter.segmentation_dataset import SegmentationDataset
from levee_hunter.augmentations import train_transform, normalize_only
from levee_hunter.paths import find_project_root, check_if_file_exists


def load_and_split(current_file_name, levees_data, data_dir, size, overlap):
    current_file_path = os.path.join(data_dir, current_file_name)

    img = rioxarray.open_rasterio(current_file_path)

    lidar_data, target = get_mask(
        img, levees_data.to_crs(img.rio.crs), invert=True, dilation_size=20
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
    db_path, levees_file_path, resolution="1m", size=1056, overlap=26
):
    import time
    import sqlite3
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import torch
    import numpy as np
    import geopandas as gpd

    if resolution not in ["1m", "13"]:
        raise ValueError("Resolution must be either '1m' or '13'.")

    tifs_path = find_project_root() / f"data/raw/w4-Lidar/{resolution}_resolution"

    # Database lock
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA locking_mode = EXCLUSIVE;")
    conn.commit()

    # Fetch unused
    unused_files = get_files_by_state(conn, "unused")

    temp_selected_parts = []
    special_selected_parts = []
    file_index = 0
    part_index = 0
    current_file_id = None
    selected_list = None
    process_active = True
    parts_dataset = None
    good_dataset = None
    bad_dataset = None

    levees_data = gpd.read_file(levees_file_path)

    output = widgets.Output()

    # Define the button variables in the outer scope:
    train_test_button = None
    validation_button = None
    keep_button = None
    special_button = None
    remove_button = None
    quit_button = None

    def create_buttons():
        nonlocal train_test_button, validation_button
        nonlocal keep_button, special_button, remove_button, quit_button

        # File selection stage
        train_test_button = widgets.Button(description="Train/Test")
        validation_button = widgets.Button(description="Validation")
        train_test_button.on_click(on_choose_train_test)
        validation_button.on_click(on_choose_validation)

        # Part selection stage
        keep_button = widgets.Button(description="Keep")
        special_button = widgets.Button(description="Special")
        remove_button = widgets.Button(description="Remove")
        keep_button.on_click(on_keep_clicked)
        special_button.on_click(on_special_clicked)
        remove_button.on_click(on_remove_clicked)

        # Quit
        quit_button = widgets.Button(description="Quit")
        quit_button.on_click(on_quit)

    def update_display():
        with output:
            clear_output(wait=True)

            if not process_active:
                print("Process has been quit. No further selections will be made.")
                return

            if file_index < len(unused_files):
                file_id, filename = unused_files[file_index]
                print(f"Processing file {file_index + 1}/{len(unused_files)}:")
                print(f"📂 File ID: {file_id}")
                print(f"📄 Filename: {filename}")

                create_buttons()

                if selected_list is None:
                    print("\nChoose an option:")
                    display(
                        widgets.VBox(
                            [train_test_button, validation_button, quit_button]
                        )
                    )
                else:
                    print(f"🆔 File ID Part {part_index + 1}/{len(parts_dataset)}")
                    parts_dataset.plot(part_index, figsize=(6, 6))
                    display(
                        widgets.VBox(
                            [keep_button, special_button, remove_button, quit_button]
                        )
                    )
            else:
                print("Finished processing all files.")

    def on_choose_train_test(b):
        nonlocal selected_list, current_file_id, part_index, parts_dataset
        selected_list = "train_test"
        with output:
            clear_output(wait=True)
            output.append_stdout("Started Processing...\n")
        current_file_id = unused_files[file_index][0]
        move_file_state(conn, current_file_id, "train_test")
        current_file_name = unused_files[file_index][1]
        output.append_stdout(f"Selected file: {current_file_name}\n")

        parts_dataset = load_and_split(
            current_file_name,
            levees_data,
            data_dir=tifs_path,
            size=size,
            overlap=overlap,
        )
        with output:
            output.append_stdout(
                f"Finished Processing. Number of image parts: {len(parts_dataset)}\n"
            )
        time.sleep(3)
        part_index = 0
        update_display()

    def on_choose_validation(b):
        nonlocal selected_list, current_file_id, part_index, parts_dataset
        selected_list = "validation"
        current_file_id = unused_files[file_index][0]
        move_file_state(conn, current_file_id, "validation")
        current_file_name = unused_files[file_index][1]
        parts_dataset = load_and_split(
            current_file_name,
            levees_data,
            data_dir=tifs_path,
            size=size,
            overlap=overlap,
        )
        part_index = 0
        update_display()

    def on_keep_clicked(b):
        nonlocal part_index
        if part_index < len(parts_dataset):
            temp_selected_parts.append(part_index)
        next_part()

    def on_special_clicked(b):
        nonlocal part_index
        if part_index < len(parts_dataset):
            temp_selected_parts.append(part_index)
            special_selected_parts.append(part_index)
        next_part()

    def on_remove_clicked(b):
        next_part()

    def on_quit(b):
        nonlocal process_active
        if selected_list is not None and current_file_id is not None:
            print(f"Reverting file {current_file_id} back to 'unused'...")
            move_file_state(conn, current_file_id, "unused")

        process_active = False
        finalize_output()

        print("Releasing database lock...")
        conn.close()

    def next_part():
        nonlocal part_index
        if not process_active:
            finalize_output()
            return
        part_index += 1
        if part_index >= len(parts_dataset):
            with output:
                clear_output(wait=True)
                output.append_stdout("Finalizing file selection...\n")
            finalize_file_selection()
        else:
            update_display()

    def finalize_file_selection():
        nonlocal selected_list, temp_selected_parts, special_selected_parts
        nonlocal good_dataset, bad_dataset
        if not process_active:
            finalize_output()
            return

        selected_set = set(temp_selected_parts)
        special_indices_set = set(special_selected_parts)
        all_indices = set(range(len(parts_dataset)))
        bad_indices = list(all_indices - selected_set)

        good_imgs = np.array([parts_dataset.images[i] for i in selected_set])
        good_targets = np.array([parts_dataset.targets[i] for i in selected_set])
        bad_imgs = np.array([parts_dataset.images[i] for i in bad_indices])
        bad_targets = np.array([parts_dataset.targets[i] for i in bad_indices])

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

        dir_path = find_project_root() / f"data/intermediate/{resolution}_{size}"
        dir_path.mkdir(parents=True, exist_ok=True)

        # Only save if there are good images (len(good_dataset) > 0)
        if selected_list == "train_test" and len(good_dataset) > 0:
            path_to_train_test = dir_path / "train_test.pth"
            if check_if_file_exists(path_to_train_test):
                old_dataset = torch.load(path_to_train_test, weights_only=False)
                if old_dataset.overlap != overlap:
                    raise ValueError("Overlap mismatch with existing dataset.")
                time.sleep(5)
                good_dataset = old_dataset + good_dataset
            torch.save(good_dataset, path_to_train_test)

        elif selected_list == "validation" and len(good_dataset) > 0:
            path_to_validation = dir_path / "validation.pth"
            if check_if_file_exists(path_to_validation):
                old_dataset = torch.load(path_to_validation, weights_only=False)
                if old_dataset.overlap != overlap:
                    raise ValueError("Overlap mismatch with existing dataset.")
                good_dataset = old_dataset + good_dataset
            torch.save(good_dataset, path_to_validation)

        # Save bad images only if there are any
        # If we try to save an empty dataset, we would try to
        # concatenate array of dimension 1 with array of dimension 3 or 4
        if len(bad_dataset) > 0:
            path_to_bad = dir_path / "bad.pth"
            if check_if_file_exists(path_to_bad):
                old_bad = torch.load(path_to_bad, weights_only=False)
                bad_dataset = old_bad + bad_dataset
            torch.save(bad_dataset, path_to_bad)

        with output:
            clear_output(wait=True)
            print(f'Saved datasets at "{dir_path}".')
        time.sleep(3)

        temp_selected_parts.clear()
        special_selected_parts.clear()
        good_dataset = None
        bad_dataset = None
        next_file()

    def next_file():
        nonlocal file_index, part_index, selected_list, current_file_id, parts_dataset
        if not process_active:
            finalize_output()
            return

        file_index += 1
        part_index = 0
        selected_list = None
        current_file_id = None
        parts_dataset = None

        if file_index < len(unused_files):
            update_display()
        else:
            finalize_output()

    def finalize_output():
        with output:
            clear_output(wait=True)
            print("Process finished.")

    # Start
    create_buttons()
    display(output)
    update_display()

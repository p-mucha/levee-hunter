import os
from pathlib import Path
import shutil
import sqlite3
import time

from levee_hunter.paths import find_project_root, check_if_file_exists
from levee_hunter.database_management import (
    add_file_to_db,
    delete_database,
    create_database,
)


def main():
    resolution = input(f"Please enter the desired resolution (1m or 13)")
    size = input(
        "Please enter the desired size of target smaller images (e.g. 256, 512)"
    )

    db_path = find_project_root() / f"data/files_db/{resolution}_{size}.db"

    if check_if_file_exists(db_path):
        print(f"WARNING: Database at {db_path} already exists. \n")
        print(f"If you proceed, the database will be deleted and recreated. \n")
        confirm_delete = input("Do you wish to continue? (yes/no): ").strip().lower()

        # if something different than yes -> cancel, otherwise delete database
        if confirm_delete != "yes":
            print("Operation canceled.")
            return

        # If user decides to proceed, delete database to reset it.
        else:
            delete_database(db_path)

    time.sleep(3)
    # This is for database creation
    # .tif files will be taken from inside this directory
    tifs_path = find_project_root() / f"data/raw/{resolution}_resolution"
    tif_files = [file for file in os.listdir(tifs_path) if file.endswith(".tif")]

    print(f"Found {len(tif_files)} .tif files in {tifs_path} \n")
    confirm_tifs = input("Do you wish to continue? (yes/no): ").strip().lower()
    if confirm_tifs != "yes":
        print("Operation canceled.")
        return

    # Create database and add files to it
    db_path.parent.mkdir(parents=True, exist_ok=True)
    create_database(db_path)
    if check_if_file_exists(db_path):
        print(f"Database created at {db_path} \n")

    # Add tif files to the database with status unused
    conn = sqlite3.connect(db_path)
    for path_to_new_file in tif_files:
        new_file_id = add_file_to_db(conn, path_to_new_file, state="unused")

    conn.close()
    print(f"Added {len(tif_files)} files to the database. \n")
    print("Database creation completed.")

    # Hardcoded target directory (must exist)
    hardcoded_dir = Path("/share/gpu5/pmucha/fathom/levee-hunter/data/files_db")
    if not hardcoded_dir.exists():
        raise FileNotFoundError(f"Hardcoded directory {hardcoded_dir} does not exist.")

    # Ask user if they want to move the database if the paths differ.
    if db_path.parent != hardcoded_dir:
        answer = (
            input(
                f"Database is currently at {db_path.parent}, which is different from the target directory ({hardcoded_dir}). "
                "Do you want to copy it to there? (yes/no): "
            )
            .strip()
            .lower()
        )
        if answer == "yes":
            new_db_path = hardcoded_dir / db_path.name
            if new_db_path.exists():
                overwrite = (
                    input(
                        f"File {new_db_path} already exists. This will replace the current file. Do you wish to continue? (yes/no): "
                    )
                    .strip()
                    .lower()
                )

                # if something different than yes -> return
                if overwrite != "yes":
                    print("Operation canceled. Database was not copied.")
                    return
                # if yes, ask for password
                else:
                    pass_1234 = input("WARNING, Please enter the password (1234): ")
                    if pass_1234 != "1234":
                        print("Wrong password. Operation canceled.")
                        return
                    else:
                        pass

            shutil.copy2(str(db_path), str(new_db_path))
            print(f"Database copied to {new_db_path}")
        else:
            print("Database was NOT copied.")


if __name__ == "__main__":
    main()

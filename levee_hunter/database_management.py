import hashlib
import sqlite3
import os


def generate_file_id_from_name(file_name, length=10):
    """
    Returns a short, deterministic ID based on the filename.
    `length` determines how many hex chars we keep from the MD5.

    Example:
        file_name = "/data/train/image01.tif"
        => "2c8a6821" (eight hex chars)
    """

    # If file_name is path, extract the filename
    filename_only = os.path.basename(file_name)

    # Compute MD5 hash of the filename (as bytes)
    md5_hash = hashlib.md5(filename_only.encode("utf-8")).hexdigest()

    # Truncate the hash to desired length
    short_id = md5_hash[:length]
    return short_id


def add_file_to_db(conn, path, state="unused"):
    """Safely adds a file to the database, preventing unintended writes when locked."""

    valid_states = ("unused", "validation", "train_test")
    if state not in valid_states:
        raise ValueError(f"Invalid state '{state}'. Must be one of {valid_states}.")

    cursor = conn.cursor()
    file_id = generate_file_id_from_name(path)  # Generate file ID

    try:
        # Insert new row
        cursor.execute(
            """
            INSERT INTO files (file_id, path, state)
            VALUES (?, ?, ?)
            """,
            (file_id, path, state),
        )

        conn.commit()  # Only commit if no errors occurred
        print(f"File {path} added successfully with state '{state}'.")
        return file_id

    except sqlite3.OperationalError as e:
        if "database is locked" in str(e):
            print(f"Database is locked. Rolling back transaction for {path}.")
            conn.rollback()  # Undo the transaction if locked
            return None  # Return None to indicate failure
        else:
            raise  # Raise any other errors


def get_file_state(conn, path):
    cursor = conn.cursor()
    cursor.execute("SELECT state FROM files WHERE path = ?", (path,))
    row = cursor.fetchone()
    if row:
        return row[0]  # e.g. "unused", "train", "validation"
    return None  # file not found


def get_files_by_state(conn, state):
    cursor = conn.cursor()
    cursor.execute("SELECT file_id, path FROM files WHERE state = ?", (state,))
    return cursor.fetchall()  # list of (file_id, path) tuples


def move_file_state(conn, identifier, new_state):
    """
    Moves the file to a new state (e.g., 'train_test', 'validation', etc.).
    'identifier' can be either a file_id or a path.
    """
    valid_states = ("unused", "validation", "train_test")
    if new_state not in valid_states:
        raise ValueError(f"Invalid state '{new_state}'. Must be one of {valid_states}.")

    cursor = conn.cursor()

    try:
        # 1. Try to interpret 'identifier' as a file_id first
        cursor.execute("SELECT file_id FROM files WHERE file_id = ?", (identifier,))
        row = cursor.fetchone()

        if row is None:
            # 2. If not found, assume 'identifier' is a path
            cursor.execute("SELECT file_id FROM files WHERE path = ?", (identifier,))
            row = cursor.fetchone()
            if row is None:
                raise ValueError(
                    f"No file found in DB matching file_id or path: {identifier}"
                )

        # row[0] is the actual file_id
        file_id = row[0]

        # 3. Perform the UPDATE for that file
        cursor.execute(
            """
            UPDATE files
            SET state = ?
            WHERE file_id = ?
            """,
            (new_state, file_id),
        )

        conn.commit()  # Only commit if no errors occurred
        print(f"File {file_id} successfully moved to '{new_state}'.")
        return file_id

    except sqlite3.OperationalError as e:
        if "database is locked" in str(e):
            print(f"Database is locked. Rolling back state change for {identifier}.")
            conn.rollback()  # Undo the transaction if locked
            return None  # Return None to indicate failure
        else:
            raise  # Raise any other errors


def count_all_files(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM files")
    return cursor.fetchone()[0]  # e.g. 42


def count_files_by_state(conn, state):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM files WHERE state = ?", (state,))
    return cursor.fetchone()[0]

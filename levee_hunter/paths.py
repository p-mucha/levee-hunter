from pathlib import Path
import string
import re
import torch
from torch import nn
from typing import Union
import warnings


def find_project_root(max_depth: int = 4):
    """Finds the root directory containing 'data', 'models' and 'levee_hunter' directories.

    Moves up in the directory structure until it finds both directories or reaches a limit.

    Args:
        max_depth (int): The maximum number of parent directories to check before stopping.

    Returns:
        Path: The project root path if found, else None.
    """
    current_path = Path.cwd()  # Start from the current directory
    home_dir = Path.home()  # User's home directory (to avoid infinite looping)

    for _ in range(max_depth):
        # Check if both 'models' and 'levee_hunter' exist in the current path
        # If they do, it means we are in the root directory
        if (
            (current_path / "models").is_dir()
            and (current_path / "levee_hunter").is_dir()
            and (current_path / "data").is_dir()
        ):
            return current_path  # Found the root directory

        # Move up one level
        if current_path == home_dir or current_path == current_path.parent:
            break  # Stop if we reach the home directory or the top level

        current_path = current_path.parent

    print("Root directory not found.")
    return None  # If no valid root is found within `max_depth`


def get_unique_model_path(base_path: Union[str, Path]) -> Path:
    """Generates a unique model path by appending A, B, C... if needed."""
    if isinstance(base_path, str):
        base_path = Path(base_path)

    base_dir = base_path.parent
    base_name = base_path.stem  # Remove .pth extension
    extension = base_path.suffix  # Get .pth extension

    # If base file doesn't exist, return it as is
    if not base_path.exists():
        return base_path

    # Get all existing model filenames in `models/`
    existing_files = [f.name for f in base_dir.glob(f"{base_name}*{extension}")]

    # Try letter-based naming (model_A.pth, model_B.pth, ...)
    for letter in string.ascii_uppercase:  # A, B, C, ..., Z
        new_path = base_dir / f"{base_name}_{letter}{extension}"
        if new_path.name not in existing_files:
            return new_path

    # If all letters A-Z are used, fallback to numbering (model_1.pth, model_2.pth, ...)
    numbers = []
    for file in existing_files:
        match = re.search(rf"{base_name}_(\d+){extension}", file)
        if match:
            numbers.append(int(match.group(1)))

    next_number = max(numbers) + 1 if numbers else 1
    return base_dir / f"{base_name}_{next_number}{extension}"


def save_model_correctly(model: nn.Module, save_model_path: Union[str, Path] = None):
    """Saves a PyTorch model, either to a user-defined path or
       an automatically generated unique path, inside the models/ directory.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        save_model_path (str or Path, optional): The file path to save the model. If None, a unique path is generated.
    """
    if save_model_path is not None:
        save_model_path = Path(save_model_path)
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_model_path)
    else:
        # Warn user that a path will be automatically generated
        warnings.warn(
            "save_model_path is None. The model will be saved to an automatically generated unique path."
        )

        # Find the project root directory
        root_dir = find_project_root()
        if root_dir is None:
            raise FileNotFoundError(
                "Could not find project root with 'models' and 'levee_hunter' directories."
            )

        # Define base model save path inside `root/models/`
        base_model_path = root_dir / "models" / f"{model.__class__.__name__}_model.pth"

        # Generate a unique filename to avoid overwriting existing models
        save_model_path = get_unique_model_path(base_model_path)

        # Ensure `models/` directory exists (create if not)
        save_model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the model to the automatically generated path
        torch.save(model.state_dict(), save_model_path)

    # Print the final save location
    print(f"Model successfully saved to: {save_model_path}")

    return save_model_path

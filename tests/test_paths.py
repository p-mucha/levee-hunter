import pytest
from pathlib import Path

from levee_hunter.paths import find_project_root


def test_print_project_root(capsys):
    # Temporarily disable output capturing so prints are shown
    with capsys.disabled():
        project_root = find_project_root()
        print(f"Found project root: {project_root}")
    # Assert that a valid path was found
    assert project_root is not None, "No project root was found."


def test_find_project_root_success(tmp_path, monkeypatch):
    """Test the find_project_root function to ensure it correctly identifies the project root.
    The find_project_root is looking for directory that contains the following 3 directories:
    - models
    - levee_hunter
    - data
    If such directory is found, we can safely assume it is the project root.
    """
    # Create a fake project structure under tmp_path.
    # For example, tmp_path / "project" will be our simulated project root.
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "models").mkdir()
    (project_root / "levee_hunter").mkdir()
    (project_root / "data").mkdir()

    # Create a nested subdirectory within the project to simulate being "inside" the project.
    nested_dir = project_root / "subdir" / "inner"
    nested_dir.mkdir(parents=True)

    # Use monkeypatch to change the current working directory to our nested directory.
    monkeypatch.chdir(nested_dir)

    # Now, when find_project_root() is called, it starts at nested_dir and moves upward.
    result = find_project_root(max_depth=10)

    # The function should find our simulated project_root.
    assert result == project_root


def test_find_project_root_failure(tmp_path, monkeypatch):
    # Create a temporary directory structure that does not match our project pattern.
    non_project = tmp_path / "non_project"
    non_project.mkdir()

    # Change the current working directory to this non-project directory.
    monkeypatch.chdir(non_project)

    # Since neither 'models' nor 'levee_hunter' exist here, find_project_root() should return None.
    result = find_project_root(max_depth=3)
    assert result is None

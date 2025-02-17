def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_splits(Z, s=256, max_overlap_frac=0.1):
    """
    Finds ways to tile dimension Z with patches of size s and overlap <= max_overlap_frac * s.

    Returns a sorted list of tuples:
        (n, overlap, total_covered)
    where:
        - n         = number of patches along one dimension
        - overlap   = integer overlap in pixels
        - total_covered = n*s - (n-1)*overlap
    """
    # Ensure s is a multiple of 32
    if s % 32 != 0:
        raise ValueError("Patch size s must be divisible by 32.")

    max_overlap_px = int(s * max_overlap_frac)
    solutions = []

    # The equation can be written as Z = n(s-o) + o
    # Which can be rearranged into n = (Z - o) // (s - o)
    # Since all are positive integers, we can get the bound by setting
    # o = 0 in the numerator and o = max_overlap_px in the denominator:
    # n <= Z // (s - max_overlap_px), then we can even add +1 to the bound
    max_n = (Z // (s - max_overlap_px)) + 1

    for n in range(1, max_n + 1):
        for overlap in range(max_overlap_px + 1):

            # Z = n*s - (n-1)*o
            total_covered = n * s - (n - 1) * overlap
            if total_covered <= Z:
                solutions.append((n, overlap, total_covered))

    # Sort primarily by how much of Z is covered (descending),
    # secondarily by overlap (ascending) so smaller overlaps appear first if coverage ties.
    solutions.sort(key=lambda x: (x[2], -x[1]), reverse=True)

    print("n, overlap, total_covered")
    return solutions


import ipywidgets as widgets
from IPython.display import display, clear_output


def interactive_labeling(dataset):
    """
    Interactive function to label images in a dataset by accepting user input.

    Parameters:
    - dataset: The dataset with a `plot(ix, figsize)` method.

    Returns:
    - selected_indices: List of indices the user selected (including special ones).
    - special_indices: Separate list of "special" indices.
    """
    selected_indices = []
    special_indices = []  # Separate list for special selections
    total_images = len(dataset)
    ix = 0  # Start index

    # UI Elements
    output = widgets.Output()
    progress_text = widgets.Label()
    button_yes = widgets.Button(description="Yes (y)")
    button_special = widgets.Button(
        description="Special (s)"
    )  # New button for special cases
    button_no = widgets.Button(description="No (n)")
    button_quit = widgets.Button(description="Quit (q)")

    def update_plot():
        """Update the displayed image and progress."""
        if ix < total_images:
            with output:
                clear_output(wait=True)
                dataset.plot(ix, figsize=(5, 5))
            progress_text.value = f"Image {ix + 1} / {total_images}"  # Show progress
        else:
            with output:
                clear_output(wait=True)
                print("Finished labeling all images.")
            progress_text.value = "Labeling complete."

    def on_yes_clicked(b):
        """Append index to selected_indices (regular selection)."""
        nonlocal ix
        selected_indices.append(ix)
        ix += 1
        update_plot()

    def on_special_clicked(b):
        """Append index to both selected_indices and special_indices."""
        nonlocal ix
        selected_indices.append(ix)
        special_indices.append(ix)  # Store in special list
        ix += 1
        update_plot()

    def on_no_clicked(b):
        """Move to the next image if user presses 'No'."""
        nonlocal ix
        ix += 1
        update_plot()

    def on_quit_clicked(b):
        """Stop the process if user presses 'Quit'."""
        with output:
            clear_output(wait=True)
            print("Labeling process stopped.")
        progress_text.value = "Process stopped."
        button_yes.disabled = True
        button_special.disabled = True
        button_no.disabled = True
        button_quit.disabled = True

    # Button Click Events
    button_yes.on_click(on_yes_clicked)
    button_special.on_click(on_special_clicked)
    button_no.on_click(on_no_clicked)
    button_quit.on_click(on_quit_clicked)

    # Display UI
    display(progress_text, output, button_yes, button_special, button_no, button_quit)
    update_plot()  # Show first image

    return selected_indices, special_indices  # Return both lists

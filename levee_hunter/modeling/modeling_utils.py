import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.draw import line as skimage_line
from skimage.morphology import skeletonize
import torch
from typing import Union, List, Tuple


def prune_skeleton(skel: np.ndarray, min_branch_length: int = 10) -> np.ndarray:
    """
    Prune skeleton branches shorter than min_branch_length.

    Inputs:
    - skel (np.ndarray): Binary skeleton (0/1). Expected shape (H, W)
    - min_branch_length (int): Minimum branch length to keep.

    Outputs:
    - np.ndarray: Pruned binary skeleton. Output shape same as input - expected (H, W).
    """
    pruned = skel.copy()

    def neighbors(y, x):
        pts = [
            (y + dy, x + dx)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dy == 0 and dx == 0)
        ]
        return [
            (ny, nx)
            for ny, nx in pts
            if 0 <= ny < pruned.shape[0]
            and 0 <= nx < pruned.shape[1]
            and pruned[ny, nx]
        ]

    def find_endpoints():
        eps = []
        for y in range(pruned.shape[0]):
            for x in range(pruned.shape[1]):
                if pruned[y, x] and len(neighbors(y, x)) == 1:
                    eps.append((y, x))
        return eps

    changed = True
    while changed:
        changed = False
        for ep in find_endpoints():
            branch = [ep]
            cur = ep
            prev = None
            # Trace the branch until a junction or non-endpoint is reached
            while True:
                nbrs = neighbors(*cur)
                if prev is not None and prev in nbrs:
                    nbrs.remove(prev)
                if len(nbrs) != 1:
                    break
                nxt = nbrs[0]
                branch.append(nxt)
                prev, cur = cur, nxt
            # If branch is too short, remove it
            if len(branch) < min_branch_length:
                # if the last point is a junction, keep it
                if len(neighbors(*branch[-1])) > 1:
                    branch = branch[:-1]
                for y, x in branch:
                    if pruned[y, x]:
                        pruned[y, x] = 0
                        changed = True
    return pruned


def get_skeleton(
    binary_mask: Union[np.ndarray, torch.Tensor],
    labeled: bool = False,
    min_branch_length: int = 10,
) -> np.ndarray:
    """
    Returns a skeleton from a binary mask (0=background, 1=foreground), with optional pruning
    of small branches shorter than min_branch_length.

    If labeled is True, returns a labeled skeleton where each connected component
    is assigned a unique label (background=0, components>=1). The default (labeled=False)
    returns a binary skeleton mask.

    Inputs:
    - binary_mask (np.ndarray or torch.Tensor): Input binary mask, (H, W) or (1, H, W) or (1, 1, H, W).
    - labeled (bool): Whether to return labeled components. Default is False.
    - min_branch_length (int): Minimum branch length to keep; branches shorter than this will be pruned.

    Outputs:
    - np.ndarray: Binary or labeled skeleton (H, W).
    """

    if isinstance(binary_mask, torch.Tensor):
        binary_mask = binary_mask.cpu().numpy()

    binary_mask = binary_mask.squeeze()

    # Ensure boolean input for skeletonize
    bool_mask = binary_mask > 0
    skel = skeletonize(bool_mask)
    skel_binary = skel.astype(np.uint8)

    # Prune small branches if min_branch_length > 0
    if min_branch_length and min_branch_length > 0.1:
        skel_binary = prune_skeleton(skel_binary, min_branch_length)

    if labeled:
        # Label connected components: background is 0, others are 1,2,...
        _, labels = cv2.connectedComponents(skel_binary)
        return labels
    else:
        return skel_binary


def find_endpoints(skel: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find endpoints in a skeleton. An endpoint is defined as a skeleton pixel
    with exactly 1 neighbor in 8 directions.

    Inputs:
    - skel (np.ndarray): Binary skeleton (0=background, 1=foreground). Expected shape (H, W).

    Outputs:
    - list of (x, y): coordinates of endpoints. Note the order x, y means column, row. Convenint for plotting.
    """
    # Label connected components first
    num_labels, labels = cv2.connectedComponents(skel.astype(np.uint8))

    endpoints = []
    for lbl in range(1, num_labels):  # label=0 is background
        component_mask = labels == lbl
        ys, xs = np.where(component_mask)

        for y, x in zip(ys, xs):
            nbrs = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]:
                        if component_mask[ny, nx]:
                            nbrs += 1
            if nbrs == 1:
                endpoints.append((y, x))

    return endpoints


def connect_close_endpoints_simple(
    skel: np.ndarray, dist_thresh: int = 20
) -> np.ndarray:
    """
    Connects any two endpoints across different connected components
    if their distance is below 'dist_thresh'.

    Inputs:
    - skel (np.ndarray): Binary skeleton (0=background, 1=foreground). Expected shape (H, W).
    - dist_thresh: max pixel distance to consider endpoints "close"

    Outputs:
    - skeleton: Updated skeleton with lines added. Expected (H, W).
    """
    skeleton = skel.copy()

    # Compute connected components and their labels
    num_labels, labels = cv2.connectedComponents(skeleton.astype(np.uint8))

    # Get endpoints from the skeleton (list of (y, x))
    endpoints = find_endpoints(skeleton)

    # Attach the label to each endpoint
    endpoints_with_labels = [(y, x, labels[y, x]) for (y, x) in endpoints]

    # Compare each pair of endpoints from different components
    for i in range(len(endpoints_with_labels)):
        y1, x1, lbl1 = endpoints_with_labels[i]
        for j in range(i + 1, len(endpoints_with_labels)):
            y2, x2, lbl2 = endpoints_with_labels[j]
            if lbl1 == lbl2:
                continue  # Skip endpoints in the same component
            dist = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if dist <= dist_thresh:
                rr, cc = skimage_line(int(y1), int(x1), int(y2), int(x2))
                skeleton[rr, cc] = 1
    return skeleton


def get_distances(
    pred: Union[np.ndarray, torch.Tensor], gt_mask: Union[np.ndarray, torch.Tensor]
):

    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()

    # Ensure binary input
    pred = (pred > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    # Ensure same shape
    assert pred.shape == gt_mask.shape, "pred and gt_mask must have the same shape"
    pred = pred.squeeze()
    gt_mask = gt_mask.squeeze()

    # Distance transform of the *complement* of the ground-truth skeleton.
    # dist_map[y, x] will be the distance from (y, x) to the nearest gt_skel==1 pixel.
    dist_map = distance_transform_edt(1 - gt_mask)

    # Extract pixel coordinates where pred_skel==1
    ys_pred, xs_pred = np.where(pred == 1)

    distances = dist_map[ys_pred, xs_pred]

    return distances


def count_pred_pixels_within_distance(
    pred: Union[np.ndarray, torch.Tensor],
    gt_mask: Union[np.ndarray, torch.Tensor],
    d: float,
) -> int:
    """
    Counts how many predicted 1 pixels are within distance d of any ground truth 1 pixel.

    Inputs:
    - pred: binary (0 for background and 1 for foreground) np.ndarray or torch.Tensor, accepted shapes: (H, W), (1, H, W), (1, 1, H, W)
    - gt_mask: binary (0 for background and 1 for foreground) np.ndarray or torch.Tensor, accepted shapes: (H, W), (1, H, W), (1, 1, H, W)
    - d: distance threshold

    Outputs:
    - int: number of predicted pixels within distance d from ground truth
    """

    distances = get_distances(pred=pred, gt_mask=gt_mask)

    # Count how many predicted pixels are within distance d
    within_distance = np.sum(distances <= d)
    return within_distance


def compute_avg_distance(
    mask1: Union[np.ndarray, torch.Tensor], mask2: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Computes average Euclidean distance from mask1 to mask2.
    Inputs:
    - mask1: binary (0 for background and 1 for foreground) np.ndarray or torch.Tensor, accepted shapes: (H, W), (1, H, W), (1, 1, H, W)
    - mask2: binary (0 for background and 1 for foreground) np.ndarray or torch.Tensor, accepted shapes: (H, W), (1, H, W), (1, 1, H, W)

    Outputs:
    - float: average distance from mask1 to mask2

    """

    if isinstance(mask1, torch.Tensor):
        mask1 = mask1.cpu().numpy()
    if isinstance(mask2, torch.Tensor):
        mask2 = mask2.cpu().numpy()

    # if both are 0 only, then we assume prediction is correct
    # so output 0
    # if one of them is 0, then we assume prediction is wrong
    # so output inf
    if np.max(mask1) == 0 and np.max(mask2) == 0:
        return 0.0
    elif np.max(mask1) == 0 or np.max(mask2) == 0:
        return np.inf

    distances = get_distances(pred=mask1, gt_mask=mask2)

    # Compute the average distance
    avg_distance = np.mean(distances)
    return avg_distance


def count_tp_fp(
    pred_skel: Union[np.ndarray, torch.Tensor],
    gt_skel: Union[np.ndarray, torch.Tensor],
    d: int,
) -> Tuple[int, int]:
    """
    Inputs:
    --------
    - pred_skel: predicted skeleton (2D array, 0 or 1).
    - gt_skel: ground truth skeleton (2D array, 0 or 1).
    - d: distance threshold.

    Outputs:
    --------
    - (tp_count, fp_count) based on distance threshold d:
    TP = # of pred_skel pixels within distance d to any gt_skel pixel
    FP = # of pred_skel pixels - TP
    """
    total_pred = int(pred_skel.sum())
    tp_count = count_pred_pixels_within_distance(pred_skel, gt_skel, d)
    fp_count = total_pred - tp_count
    return tp_count, fp_count


def count_fn(
    gt_skel: np.ndarray, pred_skel: np.ndarray, d: float, d_merge: float = None
) -> int:
    """
    Returns the number of ground truth skeleton pixels (FN) that are not
    within distance d of any predicted skeleton pixel.

    Then we also add difference in pixels between matched pixels from reconstructed
    skeletons and before reconstruction. Matched pixels are those that are within distance d of gt.

    Parameters:
        gt_skel: Ground truth skeleton (2D array, 0 or 1).
        pred_skel: Predicted skeleton (2D array, 0 or 1).
        d: Distance threshold.
        d_merge: Distance threshold for merging endpoints.

    Returns:
        int: Count of false negative pixels.
    """
    # Compute the distance transform of the complement of the predicted skeleton.
    # For each pixel, this gives the distance to the nearest pred_skel==1 pixel.
    dist_map_pred = distance_transform_edt(1 - pred_skel)

    # Get the coordinates of the ground truth skeleton pixels.
    ys_gt, xs_gt = np.where(gt_skel == 1)

    # Count how many ground truth pixels are farther than d from any predicted pixel.
    fn_count = np.sum(dist_map_pred[ys_gt, xs_gt] > d)

    if d_merge:
        merged_skel = connect_close_endpoints_simple(
            pred_skel.copy(), dist_thresh=d_merge
        )
        tp, _ = count_tp_fp(pred_skel=pred_skel, gt_skel=gt_skel, d=d)
        tp2, _ = count_tp_fp(pred_skel=merged_skel, gt_skel=gt_skel, d=d)

        fn_count += tp2 - tp

    return fn_count


def plot_skeletons(
    pred_mask: np.array, d: float, d_merge: float = None, gt_mask: np.array = None
) -> None:
    """
    Plots the masks and their skeletons.

    - If gt_mask is provided, the left plot shows the ground truth mask with its skeleton (dark blue).
    - The middle plot shows the predicted mask with its skeleton (dark blue) and endpoints (blue).
    - If d_merge is provided, the right plot shows the predicted mask with its merged skeleton (dark blue)
      overlaid with the ground truth skeleton (red) and a title with TP, FP, FN.
    - If no gt_mask is provided, TP, FP, FN are not computed and only the predicted mask is shown (with
      optionally the merged skeleton if d_merge is provided).

    Parameters:
        pred_mask: Predicted mask (binary, 0 or 1).
        d: Distance threshold (used in TP/FP/FN calculations when gt_mask is provided).
        d_merge: Distance threshold for merging endpoints.
        gt_mask: Ground truth mask (binary, 0 or 1).
    """
    # This is potentially unnecessarily long function but handles all useful cases

    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()

    if gt_mask is not None and isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()

    pred_mask = pred_mask.squeeze()
    if gt_mask is not None:
        gt_mask = gt_mask.squeeze()

    # get skeleton and endpoints of the predicted mask
    skel_pred = get_skeleton(pred_mask)
    endpoints_pred = find_endpoints(skel_pred)  # returns list of (y, x)

    # if ground truth mask is provided, compute TP, FP, FN
    if gt_mask is not None:
        skel_gt = get_skeleton(gt_mask)
        tp, fp = count_tp_fp(pred_skel=skel_pred, gt_skel=skel_gt, d=d)
        fn = count_fn(gt_skel=skel_gt, pred_skel=skel_pred, d=d, d_merge=d_merge)
    else:
        skel_gt = None
        tp, fp, fn = None, None, None

    # If d_merge is provided, get merged skeleton
    if d_merge is not None:
        skel_merged = connect_close_endpoints_simple(
            skel_pred.copy(), dist_thresh=d_merge
        )
    else:
        skel_merged = None

    # Allow up to 3 plots, depending on provided inputs
    if gt_mask is not None and d_merge is not None:
        n_plots = 3
    elif gt_mask is not None or d_merge is not None:
        n_plots = 2
    else:
        n_plots = 1

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    plot_idx = 0

    # If gt_mask is provided, plot it on the left
    # ground truth I set to red for visibility
    if gt_mask is not None:
        axes[plot_idx].imshow(gt_mask, cmap="gray")
        axes[plot_idx].set_title("Ground Truth Skeleton")
        ys, xs = np.where(skel_gt == 1)
        axes[plot_idx].scatter(xs, ys, marker=".", s=3, c="red")
        axes[plot_idx].axis("off")
        plot_idx += 1

    # Plot predicted mask with its skeleton and endpoints
    axes[plot_idx].imshow(pred_mask, cmap="gray")
    axes[plot_idx].set_title("Predicted\n(skel: blue, endpoints: orange)")
    ys, xs = np.where(skel_pred == 1)
    axes[plot_idx].scatter(xs, ys, marker=".", s=3, c="blue")
    if endpoints_pred:
        ys_ep, xs_ep = zip(*endpoints_pred)
        axes[plot_idx].scatter(xs_ep, ys_ep, marker="o", s=10, c="orange")
    axes[plot_idx].axis("off")
    plot_idx += 1

    # If d_merge is provided, plot the merged skeleton
    if d_merge is not None:
        axes[plot_idx].imshow(pred_mask, cmap="gray")
        if skel_merged is not None:
            ys_m, xs_m = np.where(skel_merged == 1)
            axes[plot_idx].scatter(xs_m, ys_m, marker=".", s=3, c="blue")
        # If ground truth is available, overlay its skeleton in red
        if skel_gt is not None:
            ys_gt, xs_gt = np.where(skel_gt == 1)
            axes[plot_idx].scatter(xs_gt, ys_gt, marker=".", s=3, c="red")
        if tp is not None and fp is not None and fn is not None:
            axes[plot_idx].set_title(
                f"GT and Merged Pred Skel\nTP={tp}, FP={fp}, FN={fn}"
            )
        else:
            axes[plot_idx].set_title("Merged")
        axes[plot_idx].axis("off")

    plt.tight_layout()
    plt.show()

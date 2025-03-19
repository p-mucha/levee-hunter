import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
import segmentation_models_pytorch as smp
from skimage.draw import line as skimage_line
from skimage.morphology import skeletonize
import torch
from typing import Tuple


def get_pred_mask(all_preds, all_targets, ix):
    pred_mask = np.array(all_preds[ix]).squeeze()
    target_mask = np.array(all_targets[ix]).squeeze()
    return pred_mask, target_mask


def prune_skeleton(skel: np.ndarray, min_branch_length: int = 10) -> np.ndarray:
    """
    Prune skeleton branches shorter than min_branch_length.

    Parameters:
        skel (np.ndarray): Binary skeleton (0/1).
        min_branch_length (int): Minimum branch length to keep.

    Returns:
        np.ndarray: Pruned binary skeleton.
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
            # If branch is too short, remove it (optionally preserving a junction)
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
    binary_mask: np.ndarray, labeled: bool = False, min_branch_length: int = 10
) -> np.ndarray:
    """
    Returns a skeleton from a binary mask (0=background, 1=foreground), with optional pruning
    of small branches shorter than min_branch_length.

    If labeled is True, returns a labeled skeleton where each connected component
    is assigned a unique label (background=0, components>=1). The default (labeled=False)
    returns a binary skeleton mask.

    Parameters:
        binary_mask (np.ndarray): Input binary mask.
        labeled (bool): Whether to return labeled components. Default is False.
        min_branch_length (int): Minimum branch length to keep; branches shorter than this will be pruned.

    Returns:
        np.ndarray: Binary or labeled skeleton.
    """
    # Ensure boolean input for skeletonize
    bool_mask = binary_mask > 0
    skel = skeletonize(bool_mask)
    skel_binary = skel.astype(np.uint8)

    # Prune small branches if min_branch_length > 0
    if min_branch_length and min_branch_length > 0:
        skel_binary = prune_skeleton(skel_binary, min_branch_length)

    if labeled:
        # Label connected components: background is 0, others are 1,2,...
        num_labels, labels = cv2.connectedComponents(skel_binary)
        return labels
    else:
        return skel_binary


def find_endpoints(skel: np.ndarray):
    """
    Find endpoints in a skeleton. An endpoint is defined as a skeleton pixel
    with exactly 1 neighbor in 8 directions.

    Returns: list of (y, x)
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


def connect_close_endpoints_simple(skel: np.ndarray, dist_thresh=20) -> np.ndarray:
    """
    Connects any two endpoints across different connected components
    if their distance is below 'dist_thresh'.

    skel: 2D skeleton (0 or 1)
    dist_thresh: max pixel distance to consider endpoints "close"

    Returns: Updated skeleton with lines added.
    """
    # Compute connected components and their labels
    num_labels, labels = cv2.connectedComponents(skel.astype(np.uint8))

    # Get endpoints from the skeleton (list of (y, x))
    endpoints = find_endpoints(skel)

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
                skel[rr, cc] = 1
    return skel


def count_pred_pixels_within_distance(
    pred_skel: np.ndarray, gt_skel: np.ndarray, d: float
) -> int:
    """
    Returns how many predicted skeleton pixels are within distance 'd'
    of any ground-truth skeleton pixel.

    pred_skel, gt_skel: 2D arrays (0 or 1)
    d: distance threshold
    """
    # Distance transform of the *complement* of the ground-truth skeleton.
    # dist_map[y, x] will be the distance from (y, x) to the nearest gt_skel==1 pixel.
    dist_map = distance_transform_edt(1 - gt_skel)

    # Extract pixel coordinates where pred_skel==1
    ys_pred, xs_pred = np.where(pred_skel == 1)

    # Count how many predicted pixels are within distance d
    within_distance = np.sum(dist_map[ys_pred, xs_pred] <= d)
    return within_distance


def count_tp_fp(pred_skel, gt_skel, d):
    """
    Returns (tp_count, fp_count) based on distance threshold d:
      TP = # of pred_skel pixels within distance d to any gt_skel pixel
      FP = # of pred_skel pixels - TP
    """
    total_pred = pred_skel.sum()
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
        axes[plot_idx].set_title("Ground Truth")
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
            axes[plot_idx].set_title(f"Merged\nTP={tp}, FP={fp}, FN={fn}")
        else:
            axes[plot_idx].set_title("Merged")
        axes[plot_idx].axis("off")

    plt.tight_layout()
    plt.show()


def get_tp_fp_fn(
    gt_mask: np.array,
    pred_mask: np.array,
    d: float,
    d_merge: float = None,
    visualise: bool = True,
) -> Tuple[int, int, int]:
    """
    Returns the number of true positives (TP), false positives (FP), and false negatives (FN)
    based on distance threshold d.

    Parameters:
        gt_mask: Ground truth mask (0 or 1).
        pred_mask: Predicted mask (0 or 1).
        d: Distance threshold.
        d_merge: Distance threshold for merging endpoints.

    Returns:
        Tuple[int, int, int]: TP, FP, FN counts.
    """
    # Compute skeletons from the masks
    skel_gt = get_skeleton(gt_mask)
    skel_pred = get_skeleton(pred_mask)

    # Count TP and FP
    tp, fp = count_tp_fp(pred_skel=skel_pred, gt_skel=skel_gt, d=d)

    # Count FN
    fn = count_fn(gt_skel=skel_gt, pred_skel=skel_pred, d=d, d_merge=d_merge)

    if visualise:
        plot_skeletons(pred_mask=pred_mask, d=d, d_merge=d_merge, gt_mask=gt_mask)

    return tp, fp, fn


def get_custom_stats(targets, predictions, d, d_merge):
    """
    Computes custom stats (TP, FP, FN) for each image in the dataset.

    Parameters:
        targets: List/array of ground truth masks.
        predictions: List/array of predicted masks.
        d: Distance threshold used in matching.
        d_merge: Merging threshold for endpoints.

    Returns:
        tp: numpy array of true positive counts for each image.
        fp: numpy array of false positive counts for each image.
        fn: numpy array of false negative counts for each image.
    """
    tp_list = []
    fp_list = []
    fn_list = []

    for i in range(len(targets)):
        pred_mask, gt_mask = get_pred_mask(predictions, targets, i)
        tp, fp, fn = get_tp_fp_fn(
            gt_mask=gt_mask, pred_mask=pred_mask, d=d, d_merge=d_merge, visualise=False
        )
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)

    # Convert lists to numpy arrays (or tensors if desired)
    tp_array = torch.tensor(tp_list).reshape(-1, 1)
    fp_array = torch.tensor(fp_list).reshape(-1, 1)
    fn_array = torch.tensor(fn_list).reshape(-1, 1)

    return tp_array, fp_array, fn_array


def custom_metrics(targets, predictions, d: int = 13, d_merge: int = 20):
    tp, fp, fn = get_custom_stats(
        targets=targets, predictions=predictions, d=d, d_merge=d_merge
    )
    tn = torch.zeros_like(tp)

    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

    # Print Results
    print("-----------------------Custom Metrics-----------------------")
    print(f"IoU Score:         {iou_score:.4f}")
    print(f"F1 Score (Dice):   {f1_score:.4f}")
    print(f"Recall:            {recall:.4f}")
    print("------------------------------------------------------------")


def standard_metrics(targets, predictions, threshold: float = 0.5):
    # Compute true positives, false positives, false negatives, true negatives
    tp, fp, fn, tn = smp.metrics.get_stats(
        predictions, targets, mode="binary", threshold=threshold
    )
    # tn = torch.zeros(3,1)
    # Compute segmentation metrics
    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

    # Print Results
    print("----------------------Standard Metrics----------------------")
    print(f"IoU Score:         {iou_score:.4f}")
    print(f"F1 Score (Dice):   {f1_score:.4f}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Recall:            {recall:.4f}")
    print("------------------------------------------------------------")

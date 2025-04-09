import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
import segmentation_models_pytorch as smp
from skimage.draw import line as skimage_line
from skimage.morphology import skeletonize
import torch
from typing import Tuple, Union


from levee_hunter.modeling.modeling_utils import (
    get_skeleton,
    plot_skeletons,
    count_tp_fp,
    count_fn,
    compute_distance_statistic,
)


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

    for ix in range(len(targets)):
        pred_mask = np.array(predictions[ix]).squeeze()
        gt_mask = np.array(targets[ix]).squeeze()
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


def custom_metrics(
    targets,
    predictions,
    d: int = 13,
    d_merge: int = 20,
    print_tp_fp_fn: bool = False,
    return_values: bool = False,
):
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

    if print_tp_fp_fn:
        print(f"TP: {tp.item()}, FP: {fp.item()}, FN: {fn.item()}")
        print("------------------------------------------------------------")

    if return_values:
        return tp, fp, fn, iou_score, f1_score, recall


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


def dist_both_ways_metric(
    mask: Union[np.ndarray, torch.Tensor],
    pred: Union[np.ndarray, torch.Tensor],
) -> float:
    dist1 = compute_distance_statistic(
        mask1=mask, mask2=pred, statistic="mean", return_length=False
    )
    dist2 = compute_distance_statistic(
        mask1=pred, mask2=mask, statistic="mean", return_length=False
    )

    return dist1 + dist2


def distance_metric(
    mask: Union[np.ndarray, torch.Tensor],
    pred: Union[np.ndarray, torch.Tensor],
) -> float:

    dist_pred_to_gt, len_pred_to_gt = compute_distance_statistic(
        mask1=pred, mask2=mask, statistic="sum", return_length=True
    )
    dist_gt_to_pred, len_gt_to_pred = compute_distance_statistic(
        mask1=mask, mask2=pred, statistic="sum", return_length=True
    )

    # if both mask and pred are 0, compute_distance_statistic will return 0
    # in both directions, we just output 0 then
    if dist_pred_to_gt == 0 and dist_gt_to_pred == 0:
        return 0.0

    if dist_pred_to_gt == np.inf or dist_gt_to_pred == np.inf:
        return np.inf

    dist_diff = np.abs(dist_gt_to_pred - dist_pred_to_gt)

    mean_dist_diff = dist_diff / np.abs(len_pred_to_gt - len_gt_to_pred)

    return mean_dist_diff + dist_gt_to_pred / (len_gt_to_pred + len_pred_to_gt + 1e-6)

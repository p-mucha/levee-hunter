import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
import rasterio
from tqdm import tqdm

import torch
from shapely.geometry import LineString
from skimage.measure import label
from skimage import measure

from levee_hunter.modeling.modeling_utils import prune_skeleton
from levee_hunter.modeling.inference import infer


def single_prediction_to_lines(
    pred, image_path,
    threshold=0.5, crs="EPSG:4269",
    min_length=3, debug_plots=False
):
    # 1. logits→probs
    if isinstance(pred, torch.Tensor):
        if pred.min()<0 or pred.max()>1:
            pred = torch.sigmoid(pred)
        pred = pred.detach().cpu().numpy()

    # 2. squeeze channel
    if pred.ndim==3 and pred.shape[0]==1:
        pred = pred[0]

    # 3. threshold → 1px mask
    binary = (pred>threshold).astype(np.uint8)

    # 4. prune skeleton
    binary = prune_skeleton(binary, min_branch_length=min_length)
    if debug_plots:
        plt.figure(figsize=(8,8))
        plt.title("Pruned Skeleton")
        plt.imshow(binary, cmap="gray")
        plt.show()
    # 4.1. label
    labeled = label(binary)

    # 5. load transform
    with rasterio.open(image_path) as src:
        transform = src.transform

    # 6. Extract ordered lines
    rows = []
    
    # Use contour tracing instead of region traversal
    contours = measure.find_contours(binary, level=0.5)

    # Filter out tiny ones
    contours = [c for c in contours if len(c) >= min_length]
    if not contours:
        # nothing to output
        return gpd.GeoDataFrame(columns=["geometry","image_path"], geometry="geometry", crs=crs)

    for contour in contours:
        # If it’s a closed loop (start and end nearly identical), drop the last point
        if np.allclose(contour[0], contour[-1], atol=1e-3):
            contour = contour[:-1]

        # Convert pixel (row, col) → world (x, y)
        geo_coords = [
            rasterio.transform.xy(transform, y, x, offset="center")
            for (y, x) in contour
        ]

        # Convert to LineString
        line = LineString(geo_coords)
        # Add to rows
        rows.append({
            "geometry": line,
            "image_path": image_path,
        })

    # debug overlay
    if debug_plots:
        img = rioxarray.open_rasterio(image_path)[0]
        fig,ax=plt.subplots(1,1,figsize=(8,8))
        img.plot.imshow(ax=ax,cmap="gray")
        gpd.GeoDataFrame(rows, crs=crs).to_crs(img.rio.crs).plot(
            ax=ax, color='red', linewidth=1
        )
        plt.title("Extracted Linesegments")
        plt.show()

    if not rows:
        return gpd.GeoDataFrame(columns=["geometry", "image_path"], geometry="geometry", crs=crs)
    else:
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def batch_predictions_to_lines(
    predictions,
    image_paths,
    threshold=0.5,
    crs="EPSG:4269",
    min_length=3,
    debug_every_n=None
):
    """
    Loop over lists of model outputs + image paths,
    extract LineStrings via single_prediction_to_lines, and
    concatenate into one GeoDataFrame (with image_path column).
    """
    all_gdfs = []
    for i, (pred, img_path) in enumerate(
            tqdm(zip(predictions, image_paths), total=len(predictions),
                 desc="Batch Extract Lines")):
        debug = (debug_every_n is not None) and (i % debug_every_n == 0)
        gdf = single_prediction_to_lines(
            pred=pred,
            image_path=img_path,
            threshold=threshold,
            crs=crs,
            min_length=min_length,
            debug_plots=debug
        )
        if not gdf.empty:
            all_gdfs.append(gdf)

    if all_gdfs:
        return gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs=crs)
    else:
        return print("No valid lines found in predictions.")


def find_matching_and_unmatched_predictions(predicted_gdf, original_gdf, buffer_meters=25):
    """
    Find both matching and unmatching predictions between predicted and original levees.
    
    Args:
        predicted_gdf: GeoDataFrame with predicted levees
        original_gdf: GeoDataFrame with original/ground truth levees
        buffer_meters: Buffer distance in meters to consider for matching
        
    Returns:
        tuple: (matching_indices, unmatched_indices)
    """
    # Ensure both GeoDataFrames have the same CRS
    if predicted_gdf.crs != original_gdf.crs:
        predicted_gdf = predicted_gdf.to_crs(original_gdf.crs)

    # Determine appropriate buffer value based on CRS
    if original_gdf.crs.is_geographic:
        # Convert meters to degrees (approximate conversion)
        buffer_value = buffer_meters / 111000  # ~0.00023 degrees per 25m
        print(f"Using buffer of {buffer_value} degrees (approximately {buffer_meters} meters)")
    else:
        # If CRS is projected (in meters), use value directly
        buffer_value = buffer_meters
        print(f"Using buffer of {buffer_value} meters")

    # Buffer the original levees to check for overlap
    buffered_originals = original_gdf.geometry.buffer(buffer_value)

    # Find both matching and non-matching predictions
    matching = []
    unmatched = []

    for idx, pred_row in tqdm(predicted_gdf.iterrows(), total=len(predicted_gdf), desc="Matching predictions"):
        # Check if this predicted levee intersects with any original levee
        intersects = any(pred_row.geometry.intersects(buff) for buff in buffered_originals)
        
        if intersects:
            matching.append(idx)
        else:
            unmatched.append(idx)

    print(f"Found {len(matching)} predicted levees that match with originals")
    print(f"Found {len(unmatched)} predicted levees that don't match with originals")

    return matching, unmatched


def plot_matching(model, device, matching, dataset, pred_gdf, original_gdf, n=3):
   # Sample n matching predictions
   if len(matching) > n:
      sample_indices = np.random.choice(matching, n, replace=False)
   else:
      sample_indices = matching
   
   # Plot n_samples prediction alongside the original data
   for i, idx in enumerate(sample_indices):
      # Get the corresponding predicted levee
      img_idx = i 
      img_filename = dataset.img_paths[img_idx]
      sample_image, sample_mask, *_ = dataset[img_idx]
      prediction = infer(model=model, image=sample_image, device=device, apply_sigmoid=True, threshold=0.5)
      pred_mask = prediction.squeeze()

      # Filter predicted GDF for this image
      pred_for_img = pred_gdf[pred_gdf["image_path"] == img_filename]

      # Load the image with rioxarray to preserve geospatial information
      img_data = rioxarray.open_rasterio(img_filename)
      pred_for_img = pred_for_img.to_crs(img_data.rio.crs)
      
      # Get the extent of the image
      left, bottom, right, top = (
         img_data.x.min().item(),
         img_data.y.min().item(),
         img_data.x.max().item(),
         img_data.y.max().item()
      )
      
      # Create a figure with two subplots
      fig, axes = plt.subplots(1, 3, figsize=(15, 7))
      
      for ax in axes:
         ax.set_xlim([left, right])
         ax.set_ylim([bottom, top])
         ax.set_aspect('equal')   
      
      # Plot the image and the predicted levee from the tensor array
      img_data[0].plot.imshow(ax=axes[0], cmap='gray')
      axes[0].set_title(f'Original Prediction')
      # Plot the prediction as an overlay on the image
      axes[0].imshow(pred_mask, alpha=0.5, cmap='OrRd', extent=[left, right, bottom, top])
      
      # Plot the image on the left w/ predicted levee
      img_data[0].plot.imshow(ax=axes[1], cmap='gray')
      axes[1].set_title(f'Levee in GeoDataFrame')
      pred_for_img.plot(ax=axes[1], color='red', linewidth=1, label='Predicted Levee(s)')
      axes[1].legend()

      # Plot the original and predicted levees on the right
      img_data[0].plot.imshow(ax=axes[2], cmap='gray')
      axes[2].set_title('Original vs Predicted Levee(s)')
      pred_for_img.plot(ax=axes[2], color='red', linewidth=1, label='Predicted Levee(s)')
      original_gdf.plot(ax=axes[2], color='blue', linewidth=1, label='Original Levee(s)')
      axes[2].legend()

      plt.tight_layout()
      plt.show()


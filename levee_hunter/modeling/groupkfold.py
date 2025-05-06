import numpy as np
import pandas as pd
import re
import random
import tqdm

from torch.utils.data import DataLoader, SubsetRandomSampler
from levee_hunter.modeling.train import train_model
from levee_hunter.modeling.inference import get_preds_targets
from levee_hunter.modeling.modeling_utils import compute_distance_statistic
from sklearn.model_selection import GroupKFold

def dice_score(y_true, y_pred, smooth=1e-6):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    intersection = (y_true * y_pred).sum()
    return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

def iou_score(y_true, y_pred, smooth=1e-6):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def GroupKFoldTraining(dataset, 
                       device,
                       model,
                       optimizer,
                       criterion,
                       model_path_output,
                       num_folds=5,
                       epochs=10,
                       batch_size=16,
                       ):
   
   """
   Perform Group K-Fold cross-validation training on a dataset.
   
   Args:
       dataset: The dataset to train on.
       device: The device to use for training (CPU or GPU).
       model: The model to train.
       optimizer: The optimizer to use for training.
       model_path_output: The path to save the trained model.
       num_folds: The number of folds for cross-validation.
       epochs: The number of epochs to train for each fold.
       
   Returns:
       fold_results: A list of dictionaries containing metrics for each fold.
       all_train_losses: A list of training losses for each fold.
       all_val_losses: A list of validation losses for each fold.
       per_image_results: A DataFrame containing per-image results for each fold.
   """

   # Create a GroupKFold object
   kfold = GroupKFold(n_splits=num_folds)

   # Lists to store metrics for each fold
   fold_results = []
   all_train_losses = []
   all_val_losses = []
   per_image_results = []  # To store per-image results

   # Create a list of all dataset indices
   dataset_size = len(dataset)
   indices = list(range(dataset_size))

   batch_size = batch_size

   # Get group IDs for the dataset
   groups = group_ids(dataset, print_info=False)

   for fold, (train_idx, val_idx) in enumerate(kfold.split(indices, groups=groups)):
      print(f"Starting fold {fold+1}/{num_folds}")
      
      # Create train and validation subdatasets
      train_subsampler = SubsetRandomSampler(train_idx)
      val_subsampler = SubsetRandomSampler(val_idx)
      
      # Define data loaders for this fold
      fold_train_loader = DataLoader(
         dataset, 
         batch_size=batch_size, 
         sampler=train_subsampler,
         shuffle=False
      )
      
      fold_val_loader = DataLoader(
         dataset,
         batch_size=batch_size,
         sampler=val_subsampler,
         shuffle=False
      )
      
      # Reset model for each fold
      model_current = model.to(device)
      optimizer = optimizer

      # Define model name with fold number to save each fold's model separately
      model_path = f"{model_path_output}_{fold+1}.pth"

      model_current, train_losses, val_losses = train_model(
         model=model_current,
         train_loader=fold_train_loader,
         test_loader=fold_val_loader,
         optimizer=optimizer,
         criterion=criterion,
         epochs=epochs,
         suppress_output=True,
         save_model='best',
         save_model_path=model_path,
      )
      
      # Save the losses
      all_train_losses.append(train_losses)
      all_val_losses.append(val_losses)
      
      # Evaluate the model
      model_current.eval()
      preds, targets = get_preds_targets(model=model_current, val_loader=fold_val_loader, device=device)

      # Get the actual image indices from the validation sampler
      # We need to map back to the original dataset indices
      val_indices = [val_idx[i] for i in range(len(preds))]
      # Get file names for the validation images
      val_filenames = [dataset.img_paths[idx] for idx in val_indices]
      
      image_metrics = []
      for i in range(len(preds)):
         pred = preds[i]
         target = targets[i]
         
         # Calculate metrics for this image
         dice = dice_score(target, pred)
         iou = dice_score(target, pred)
         distance = compute_distance_statistic(pred, target)
         
         image_metrics.append({
            'dice': dice,
            'iou': iou,
            'distance': distance,
            'filename': val_filenames[i]
         })
      
      # Convert to DataFrame
      fold_per_image_results = pd.DataFrame(image_metrics)
      fold_per_image_results['fold'] = fold+1
      per_image_results.append(fold_per_image_results)
      
      # Calculate overall metrics
      print("Calculating overall metrics for this fold...")
      avg_dice = np.mean([m['dice'] for m in image_metrics])
      avg_iou = np.mean([m['iou'] for m in image_metrics])
      avg_distance = np.mean([m['distance'] for m in image_metrics])
      
      # Create metrics dictionary for this fold
      fold_metrics = {
         'fold': fold + 1,
         'dice': avg_dice,
         'iou': avg_iou,
         'distance': avg_distance,
         'val_loss': val_losses[-1]  # Final validation loss
      }
      
      # Append to results list
      fold_results.append(fold_metrics)
      
      print(f"\nFold {fold+1} metrics: \nDice={avg_dice:.4f}, \nIoU={avg_iou:.4f}, \nDistance={avg_distance:.4f}, \nVal Loss={val_losses[-1]:.4f}\n\n")

   return fold_results, all_train_losses, all_val_losses, per_image_results


def group_ids(dataset, print_info=True):
   """
   Generate group IDs based on the filenames in the dataset.
   Args:
       dataset: The dataset to generate group IDs for.
   Returns:
       group_ids: A list of group IDs corresponding to the dataset images.
   """

   # Extract group IDs from the file paths - keeping everything before "_pX_wY"
   groups = []
   for path in dataset.img_paths:
      # Convert path to string if it's a Path object
      path_str = str(path)
      # Extract the base pattern without "_pX_wY.tif"
      match = re.search(r'(.+)_p\d+_w\d+\.tif$', path_str)
      if match:
         group_id = match.group(1)
         groups.append(group_id)
      else:
         # Fallback if pattern doesn't match
         groups.append(path_str)

   if print_info:
      print(f"Total images: {len(groups)}")
      print(f"Unique groups: {len(set(groups))}")

   unique_groups = list(set(groups))
   random.shuffle(unique_groups)
   group_to_idx = {g: i for i, g in enumerate(unique_groups)}
   group_ids = [group_to_idx[g] for g in groups]
   return group_ids
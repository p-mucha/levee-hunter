import torch
import torch.nn as nn

from levee_hunter.segmentation_dataset import SegmentationDataset
from levee_hunter.paths import save_model_correctly


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion=nn.BCEWithLogitsLoss(reduction="none"),
    epochs=10,
    save_model=True,
    save_model_path=None,
):
    if not isinstance(train_loader.dataset, SegmentationDataset):
        raise ValueError("train_loader.dataset must be a SegmentationDataset instance")

    if not isinstance(val_loader.dataset, SegmentationDataset):
        raise ValueError("val_loader.dataset must be a SegmentationDataset instance")

    best_loss = float("inf")

    train_loss_list = []
    val_loss_list = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("beginning training on device:", device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            if train_loader.dataset.weights_return:
                images, mask, weights = batch  # Extract weights if present
                images, mask, weights = (
                    images.to(device),
                    mask.to(device),
                    weights.to(device),
                )
            else:
                images, mask = batch
                images, mask = images.to(device), mask.to(device)
                weights = torch.ones(
                    mask.shape[0], device=device
                )  # Default weights = 1 if not provided

            optimizer.zero_grad()
            output = model(images)

            # Compute weighted loss
            # By default reduction is mean, so it would take an average over
            # all images and all pixels in the batch, returning single scalar
            # But that would make it impossible to apply weights
            # So in our case we set reduction to none, this creates an instance of
            # loss function which is then applied to output, mask, where output is actually
            # an array of outputs (batch_size, 1, H, W)
            loss = criterion(output, mask)  # Compute per-element loss

            # Computing weighted loss is just computing a weighted average
            # We normally would do loss.mean() which gets an average over
            # all images in batch and all their pixels
            # Here we want to still take an average over pixels, but
            # the average over images should be weighted
            # loss.shape = (batch_size, 1, H, W), so to average over pixels
            # we can take mean with dim=(2,3) argument, this results in
            # loss.shapr = (batch_size, 1), so to multiply by weights we need to
            # reshape with .view(-1, 1). Then to get weighted average we sum the loss
            # and divide over sum of weights
            # Loss is then just a scalar for which we can then compute gradients with
            # respect to model parameters
            loss = loss.mean(
                dim=(2, 3)
            )  # Compute mean over H and W (pixel-wise mean per image)

            # Normalize by total weight sum (to match validation loss scale)
            loss = (loss * weights.view(-1, 1)).sum() / weights.sum()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, mask = batch  # Validation does not use weights
                images, mask = images.to(device), mask.to(device)

                output = model(images)
                loss = criterion(output, mask)  # Standard loss for validation

                loss = loss.mean()  # as we use reduction='none'

                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(
            f"Epoch: {epoch+1}/{epochs} Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f}"
        )

        if save_model:
            if val_loss < best_loss:
                best_loss = val_loss

                # Save model with best validation loss
                # This also updates the save_model_path variable, so that
                # there are multiple models saved (model_A, model_B, ...), during
                # single training run
                save_model_path = save_model_correctly(model, save_model_path)

    model_architecture = model.__class__.__name__
    encoder_name = model.encoder.__class__.__name__

    print(f"Trained {model_architecture} model with {encoder_name} encoder.")

    return model, train_loss_list, val_loss_list

import torch
import torch.nn as nn

from levee_hunter.levees_dataset import LeveesDataset
from levee_hunter.paths import save_model_correctly

def calculate_levee_vs_nonlevee_loss(outputs, masks, criterion):
    """Calculate separate loss metrics for levee and non-levee images."""
    # Outputs and masks should be on the same device
    device = outputs.device
    
    # Determine which images have levees (at least one positive pixel in mask)
    has_levee = (masks.sum(dim=(1, 2, 3)) > 0).float()
    
    # Calculate loss for each image
    per_image_loss = criterion(outputs, masks)
    if len(per_image_loss.shape) == 4:
        per_image_loss = per_image_loss.mean(dim=(2, 3))  # Average over pixels
    
    # Separate losses for levee and non-levee images
    levee_loss = torch.zeros(1, device=device)
    nonlevee_loss = torch.zeros(1, device=device)
    
    # Count of each type of image
    levee_count = has_levee.sum().item()
    nonlevee_count = (1 - has_levee).sum().item()
    
    # Calculate average loss for each type if they exist
    if levee_count > 0:
        levee_loss = (per_image_loss * has_levee.view(-1, 1)).sum() / levee_count
    
    if nonlevee_count > 0:
        nonlevee_loss = (per_image_loss * (1 - has_levee).view(-1, 1)).sum() / nonlevee_count
    
    return levee_loss.item(), nonlevee_loss.item(), levee_count, nonlevee_count


def train_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion=nn.BCEWithLogitsLoss(reduction="none"),
    epochs=10,
    save_model="best",
    save_model_path=None,
):
    if not isinstance(train_loader.dataset, LeveesDataset):
        raise ValueError("train_loader.dataset must be a LeveesDataset instance")

    if not isinstance(test_loader.dataset, LeveesDataset):
        raise ValueError("test_loader.dataset must be a LeveesDataset instance")

    if save_model not in ["best", "last"]:
        raise ValueError("save_model must be either 'best' or 'last'")

    best_loss = float("inf")

    train_loss_list = []
    test_loss_list = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("beginning training on device:", device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_levee_loss = 0.0
        train_nonlevee_loss = 0.0
        train_levee_count = 0
        train_nonlevee_count = 0

        for batch in train_loader:
            if train_loader.dataset.weighted:
                images, mask, weights = batch  # Extract weights if present
                images, mask, weights = (
                    images.to(device),
                    mask.to(device),
                    weights.to(device),
                )
            else:
                images, mask = batch[:2]
                images, mask = images.to(device), mask.to(device)
                weights = torch.ones(
                    mask.shape[0], device=device
                )  # Default weights = 1 if not provided

            optimizer.zero_grad()
            output = model(images)

            # Compute weighted loss
            loss = criterion(output, mask)  # Compute per-element loss

            if len(loss.shape) == 4:
                loss = loss.mean(
                    dim=(2, 3)
                )  # Compute mean over H and W (pixel-wise mean per image)

            # Normalize by total weight sum (to match testing loss scale)
            loss = (loss * weights.view(-1, 1)).sum() / weights.sum()

            # Calculate separate metrics for levee and non-levee images
            batch_levee_loss, batch_nonlevee_loss, levee_count, nonlevee_count = calculate_levee_vs_nonlevee_loss(output, mask, criterion)
            train_levee_loss += batch_levee_loss * levee_count
            train_nonlevee_loss += batch_nonlevee_loss * nonlevee_count
            train_levee_count += levee_count
            train_nonlevee_count += nonlevee_count

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        total_samples = 0
        test_loss = 0.0
        test_levee_loss = 0.0
        test_nonlevee_loss = 0.0
        test_levee_count = 0
        test_nonlevee_count = 0

        with torch.no_grad():
            for batch in test_loader:
                images, mask = batch[:2]  # Testing does not use weights
                images, mask = images.to(device), mask.to(device)

                output = model(images)
                loss = criterion(output, mask)  # Standard loss for testing

                if len(loss.shape) == 4:
                    # loss shape is (batch_size, 1, H, W); average over pixels per image
                    loss = loss.mean(dim=(2, 3))  # now shape (batch_size, 1)

                batch_size = images.shape[0]
                total_samples += batch_size
                test_loss += loss.sum().item()

                # Calculate separate metrics for levee and non-levee images
                batch_levee_loss, batch_nonlevee_loss, levee_count, nonlevee_count = calculate_levee_vs_nonlevee_loss(output, mask, criterion)
                test_levee_loss += batch_levee_loss * levee_count
                test_nonlevee_loss += batch_nonlevee_loss * nonlevee_count
                test_levee_count += levee_count
                test_nonlevee_count += nonlevee_count

        train_loss /= len(train_loader)
        test_loss /= total_samples

        # Calculate average losses for levee and non-levee images
        train_levee_loss = train_levee_loss / max(train_levee_count, 1)
        train_nonlevee_loss = train_nonlevee_loss / max(train_nonlevee_count, 1)
        test_levee_loss = test_levee_loss / max(test_levee_count, 1)
        test_nonlevee_loss = test_nonlevee_loss / max(test_nonlevee_count, 1)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        print(
            f"Epoch: {epoch+1}/{epochs} Train Loss: {train_loss:.6f} Test Loss: {test_loss:.6f}"
        )
        print(
            f"Train Levee Loss: {train_levee_loss:.6f} ({train_levee_count}) "
            f"Train Non-Levee Loss: {train_nonlevee_loss:.6f} ({train_nonlevee_count})"
        )
        print(
            f"Test Levee Loss: {test_levee_loss:.6f} ({test_levee_count}) "
            f"Test Non-Levee Loss: {test_nonlevee_loss:.6f} ({test_nonlevee_count})"
        )

        if save_model == "best":
            if test_loss < best_loss:
                best_loss = test_loss

                # Save model with best testing loss
                save_model_path = save_model_correctly(model, save_model_path)

    if save_model == "last":
        save_model_path = save_model_correctly(model, save_model_path)

    model_architecture = model.__class__.__name__
    encoder_name = model.encoder.__class__.__name__

    print(f"Trained {model_architecture} model with {encoder_name} encoder.")

    return model, train_loss_list, test_loss_list

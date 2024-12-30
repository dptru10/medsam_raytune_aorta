import os
import tempfile
import nrrd
import torch
import copy
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from torchvision import transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.train import Checkpoint

# Set up environment variables
os.environ["DATA_DIRECTORY"] = "/home/dennist/Completed_Mask_Volume_Folders"
os.environ["IMG_DIRECTORY"] = "/home/dennist/AortaSegmentation/training_results_raytune/figs"
data_directory = os.environ["DATA_DIRECTORY"]
figures_dir = os.environ["IMG_DIRECTORY"]
os.makedirs(figures_dir, exist_ok=True)

def load_nrrd_file(file_path):
    """Load an NRRD file and return the data."""
    data, header = nrrd.read(file_path)
    return data, header

def get_representative_slices(start, end):
    if start == end:
        return [start, start, start]
    total_slices = end - start + 1
    return [
        start,
        start + total_slices // 2,
        end
    ]

def get_slice(data, slice_index, view="Axial"):
    """Get the appropriate slice based on the selected view."""
    if view == "Axial":
        return data[:, :, slice_index]
    elif view == "Coronal":
        return data[:, slice_index, :]
    elif view == "Sagittal":
        return data[slice_index, :, :]

def transform_slice(slice_image, view="Axial"):
    """Apply necessary transformations to the slice."""
    # Rotate the image 90 degrees clockwise
    slice_image = np.rot90(slice_image, k=-1)
    # Flip the image from left to right along the vertical axis
    slice_image = np.fliplr(slice_image)
    # Flip the image from top to bottom along the horizontal axis for coronal and sagittal views
    if view in ["Coronal", "Sagittal"]:
        slice_image = np.flipud(slice_image)
    return slice_image

def read_nrrd_files(data_path, data_type):
    folders = [folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))]
    data_list = []

    for folder in folders:
        try:
            folder_path = os.path.join(data_path, folder)
            nrrd_path = os.path.join(folder_path, 'volume' if data_type == 'volume' else 'mask')
            nrrd_files = [file for file in os.listdir(nrrd_path) if file.endswith('.nrrd')]
            print(f"Reading {data_type} from {folder}")

            if len(nrrd_files) != 1:
                print(f"Skipping {folder}: Expected 1 NRRD file, found {len(nrrd_files)} in {nrrd_path}")
                continue

            file_path = os.path.join(nrrd_path, nrrd_files[0])
            data, _ = load_nrrd_file(file_path)
            data_list.append(data)

        except FileNotFoundError as e:
            print(f"Skipping {folder}: {str(e)}")
        except Exception as e:
            print(f"Error processing {folder}: {str(e)}")

    return data_list

def find_nonzero_range(mask_data):
    """Find the range of non-zero slices in the mask."""
    nonzero_slices = np.where(np.any(mask_data, axis=(0, 1)))[0]
    if len(nonzero_slices) == 0:
        return None, None
    return nonzero_slices[0], nonzero_slices[-1]

class AortaDataset2D(Dataset):
    def __init__(self, volumes, masks):
        self.data = []
        
        # Define transforms for image and mask
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((1024, 1024), antialias=True),
            transforms.ToTensor()
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
        for volume, mask in zip(volumes, masks):
            start_slice, end_slice = find_nonzero_range(mask)
            if start_slice is None or end_slice is None:
                continue
            
            for i in range(start_slice, end_slice + 1):
                volume_slice = transform_slice(get_slice(volume, i, "Axial"), "Axial")
                mask_slice = transform_slice(get_slice(mask, i, "Axial"), "Axial")
                
                # Normalize volume slice to [0, 1]
                if volume_slice.max() != volume_slice.min():
                    volume_slice = (volume_slice - volume_slice.min()) / (volume_slice.max() - volume_slice.min())
                
                # Convert to torch tensors and apply transforms
                volume_slice = (volume_slice * 255).astype(np.uint8)  # Convert to uint8 for PIL
                mask_slice = (mask_slice * 255).astype(np.uint8)
                
                volume_slice = self.image_transform(volume_slice)  # Will be (1, 1024, 1024)
                mask_slice = self.mask_transform(mask_slice)  # Will be (1, 1024, 1024)
                
                # Repeat the channel dimension for volume slice
                volume_slice = volume_slice.repeat(3, 1, 1)  # Now (3, 1024, 1024)
                
                # Convert mask back to binary
                mask_slice = (mask_slice > 0.5).float()
                
                # Get bounding box from mask
                y_indices, x_indices = torch.where(mask_slice[0] > 0)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    x_min, x_max = x_indices.min().item(), x_indices.max().item()
                    y_min, y_max = y_indices.min().item(), y_indices.max().item()
                    bbox = np.array([x_min, y_min, x_max, y_max])  # Store as numpy array
                else:
                    continue
                
                self.data.append({
                    'image': volume_slice,  # Already in CHW format (3, 1024, 1024)
                    'label': mask_slice[0],  # (1024, 1024)
                    'bbox': bbox  # (4,) numpy array
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def visualize_results(model, test_volume, test_mask, slice_indices, device, output_dir):
    slice_names = ['Upper Third', 'Middle', 'Lower Third']
    
    # Define transforms
    image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1024, 1024), antialias=True),
        transforms.ToTensor()
    ])
    
    mask_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(20, 20))
    plt.suptitle("Inference Results (Axial View)", fontsize=16)

    for idx, (slice_index, slice_name) in enumerate(zip(slice_indices, slice_names)):
        test_slice = transform_slice(get_slice(test_volume, slice_index, "Axial"), "Axial")
        ground_truth = transform_slice(get_slice(test_mask, slice_index, "Axial"), "Axial")
        
        # Normalize and convert to uint8
        if test_slice.max() != test_slice.min():
            test_slice = (test_slice - test_slice.min()) / (test_slice.max() - test_slice.min())
        test_slice = (test_slice * 255).astype(np.uint8)
        ground_truth = (ground_truth * 255).astype(np.uint8)
        
        # Apply transforms
        test_slice = image_transform(test_slice)  # Will be (1, 1024, 1024)
        ground_truth = mask_transform(ground_truth)  # Will be (1, 1024, 1024)
        
        # Convert mask back to binary
        ground_truth = (ground_truth > 0.5).float()
        
        # Repeat channels for test slice
        test_slice = test_slice.repeat(3, 1, 1)  # Now (3, 1024, 1024)
        
        # Get bounding box from ground truth
        y_indices, x_indices = torch.where(ground_truth[0] > 0)
        if len(x_indices) > 0 and len(y_indices) > 0:
            x_min, x_max = x_indices.min().item(), x_indices.max().item()
            y_min, y_max = y_indices.min().item(), y_indices.max().item()
            bbox = np.array([[x_min, y_min, x_max, y_max]])
        else:
            continue

        # Prepare input for model
        model_input = test_slice.unsqueeze(0).float().to(device)

        # Perform inference
        with torch.no_grad():
            predicted_mask = model(model_input, bbox).squeeze().sigmoid().cpu()
        predicted_mask = (predicted_mask > 0.5).float()

        # Convert tensors to numpy for plotting
        test_slice_np = test_slice[0].numpy()
        ground_truth_np = ground_truth[0].numpy()
        predicted_mask_np = predicted_mask.numpy()

        row = idx + 1

        plt.subplot(3, 4, row * 4 - 3)
        plt.title(f"{slice_name} Slice - Original Image")
        plt.imshow(test_slice_np, cmap="gray")
        plt.axis('off')

        plt.subplot(3, 4, row * 4 - 2)
        plt.title(f"{slice_name} Slice - Ground Truth")
        plt.imshow(ground_truth_np, cmap="gray")
        plt.axis('off')

        plt.subplot(3, 4, row * 4 - 1)
        plt.title(f"{slice_name} Slice - Predicted Mask")
        plt.imshow(predicted_mask_np, cmap="gray")
        plt.axis('off')

        plt.subplot(3, 4, row * 4)
        plt.title(f"{slice_name} Slice - Overlay")
        plt.imshow(test_slice_np, cmap="gray")
        plt.imshow(predicted_mask_np, alpha=0.5, cmap="jet")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'inference_example_{current_date}.png'))
    plt.close()

class MedSAM(nn.Module):
    def __init__(self, sam_model):
        super().__init__()
        self.image_encoder = sam_model.image_encoder
        self.mask_decoder = sam_model.mask_decoder
        self.prompt_encoder = sam_model.prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)
        
        with torch.no_grad():
            if isinstance(box, np.ndarray):
                box = torch.from_numpy(box)
            
            # Ensure box is of shape (B, 1, 4)
            if len(box.shape) == 2:
                box = box.unsqueeze(1)
            
            box_torch = box.to(image.device, dtype=torch.float32)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        
        return ori_res_masks

def train_model(config):
    # Load data
    data_path = os.environ["DATA_DIRECTORY"]
    volumes = read_nrrd_files(data_path, 'volume')
    masks = read_nrrd_files(data_path, 'mask')

    if not volumes or not masks:
        print("Error: No valid data found. Exiting.")
        return

    # Split data and create datasets
    train_volumes, val_volumes, train_masks, val_masks = train_test_split(
        volumes, masks, test_size=0.2, random_state=42
    )

    # Create datasets with data augmentation based on config
    train_dataset = AortaDataset2D(train_volumes, train_masks)
    val_dataset = AortaDataset2D(val_volumes, val_masks)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize the SAM model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_model = sam_model_registry[config["model_type"]](checkpoint=config["checkpoint"])
    
    # Create MedSAM model
    model = MedSAM(sam_model).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Set up optimizer and loss function
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True)

    best_dice = 0.0

    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0
        step = 0
        dice_metric = DiceMetric(include_background=False, reduction="mean")

        for batch_data in train_loader:
            step += 1
            inputs = batch_data['image'].to(device).float()
            labels = batch_data['label'].unsqueeze(1).to(device).float()
            bboxes = batch_data['bbox']
            
            if isinstance(bboxes, list):
                bboxes = torch.tensor(bboxes, device=device)
            else:
                bboxes = bboxes.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, bboxes)
            loss = loss_function(outputs, labels) + F.binary_cross_entropy_with_logits(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = (outputs > 0.5).float()
            dice_metric(y_pred=pred, y=labels)

        epoch_loss /= step
        train_dice = dice_metric.aggregate().item()
        dice_metric.reset()

        # Validation
        model.eval()
        val_loss = 0
        val_step = 0
        val_dice_metric = DiceMetric(include_background=False, reduction="mean")

        with torch.no_grad():
            for val_data in val_loader:
                val_step += 1
                val_inputs = val_data['image'].to(device).float()
                val_labels = val_data['label'].unsqueeze(1).to(device).float()
                val_bboxes = val_data['bbox']
                
                if isinstance(val_bboxes, list):
                    val_bboxes = torch.tensor(val_bboxes, device=device)
                else:
                    val_bboxes = val_bboxes.to(device)

                val_outputs = model(val_inputs, val_bboxes)
                val_loss += (loss_function(val_outputs, val_labels) + 
                           F.binary_cross_entropy_with_logits(val_outputs, val_labels)).item()

                val_pred = (val_outputs > 0.5).float()
                val_dice_metric(y_pred=val_pred, y=val_labels)

        val_loss /= val_step
        val_dice = val_dice_metric.aggregate().item()
        val_dice_metric.reset()

        # Report metrics
        metrics = {
            "train_loss": epoch_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_dice": val_dice
        }
        
        if val_dice > best_dice:
            best_dice = val_dice
            # Save checkpoint
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
                torch.save(model.state_dict(), checkpoint_path)
                session.report(metrics, checkpoint=Checkpoint.from_directory(tmpdir))
        else:
            session.report(metrics)

def main():
    # Define the hyperparameter search space
    config = {
        "model_type": "vit_b",
        "checkpoint": "work_dir/SAM/sam_vit_b_01ec64.pth",
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "num_epochs": 10
    }

    scheduler = ASHAScheduler(
        metric="val_dice",
        mode="max",
        max_t=10,
        grace_period=2,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size", "weight_decay"],
        metric_columns=["train_loss", "val_loss", "train_dice", "val_dice", "training_iteration"])

    storage_path = '/home/dennist/AortaSegmentation/training_results_raytune/raytune_logs'
    
    result = tune.run(
        train_model,
        resources_per_trial={"cpu": 4, "gpu": 1 if torch.cuda.is_available() else 0},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter,
        storage_path=storage_path,
        log_to_file=True)

    # Get best trial
    best_trial = result.get_best_trial("val_dice", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["val_loss"]))
    print("Best trial final validation dice: {}".format(
        best_trial.last_result["val_dice"]))

    # Load the best checkpoint and evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_model = sam_model_registry[best_trial.config["model_type"]](
        checkpoint=best_trial.config["checkpoint"])
    best_trained_model = MedSAM(sam_model).to(device)

    best_checkpoint = best_trial.checkpoint
    with best_checkpoint.as_directory() as checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        best_trained_model.load_state_dict(torch.load(checkpoint_path))
# Load test data
    data_path = os.environ["DATA_DIRECTORY"]
    volumes = read_nrrd_files(data_path, 'volume')
    masks = read_nrrd_files(data_path, 'mask')

    train_volumes, val_volumes, train_masks, val_masks = train_test_split(
        volumes, masks, test_size=0.2, random_state=42
    )

    # Create validation dataset
    val_dataset = AortaDataset2D(val_volumes, val_masks)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Evaluate on validation set
    best_trained_model.eval()
    val_dice_metric = DiceMetric(include_background=False, reduction="mean")
    val_losses = []

    with torch.no_grad():
        for val_data in tqdm(val_loader, desc="Final Evaluation"):
            val_inputs = val_data['image'].to(device).float()
            val_labels = val_data['label'].unsqueeze(1).to(device).float()
            val_bboxes = val_data['bbox']
            
            if isinstance(val_bboxes, list):
                val_bboxes = torch.tensor(val_bboxes, device=device)
            else:
                val_bboxes = val_bboxes.to(device)

            val_outputs = best_trained_model(val_inputs, val_bboxes)
            val_loss = DiceCELoss(to_onehot_y=False, sigmoid=True)(val_outputs, val_labels)
            val_losses.append(val_loss.item())

            val_pred = (val_outputs > 0.5).float()
            val_dice_metric(y_pred=val_pred, y=val_labels)

    final_val_dice = val_dice_metric.aggregate().item()
    final_val_loss = np.mean(val_losses)
    
    print(f"Final Test Results:")
    print(f"Mean Dice Score: {final_val_dice:.4f}")
    print(f"Mean Loss: {final_val_loss:.4f}")

    # Visualization of results
    test_volume = val_volumes[-1]  # Use last validation volume for visualization
    test_mask = val_masks[-1]
    
    # Find non-zero range for visualization
    start_slice, end_slice = find_nonzero_range(test_mask)
    if start_slice is not None and end_slice is not None:
        slice_indices = get_representative_slices(start_slice, end_slice)
        visualize_results(best_trained_model, test_volume, test_mask, slice_indices, device, figures_dir)
        print(f"Visualization saved to {figures_dir}")
    else:
        print("No non-zero slices found in the test mask. Skipping visualization.")

    # Save final results to a file
    results_file = os.path.join(figures_dir, f'results_{datetime.now().strftime("%Y%m%d_%H%M")}.txt')
    with open(results_file, 'w') as f:
        f.write(f"Best Trial Config:\n{best_trial.config}\n\n")
        f.write(f"Final Test Results:\n")
        f.write(f"Mean Dice Score: {final_val_dice:.4f}\n")
        f.write(f"Mean Loss: {final_val_loss:.4f}\n")

    # Optional: Save the best model
    model_save_path = os.path.join(figures_dir, f'best_model_{datetime.now().strftime("%Y%m%d_%H%M")}.pth')
    torch.save({
        'model_state_dict': best_trained_model.state_dict(),
        'config': best_trial.config,
        'final_dice': final_val_dice,
        'final_loss': final_val_loss
    }, model_save_path)
    print(f"Best model saved to {model_save_path}")

if __name__ == "__main__":
    main()
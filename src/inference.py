# inference.py

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Local imports
from dataset import (
    read_nrrd_files, transform_slice, get_slice, find_nonzero_range,
    get_representative_slices
)
from model import MedSAM
from segment_anything import sam_model_registry


def visualize_inference_results(
    model,
    volume,
    mask,
    slice_indices,
    device,
    output_dir
):
    """
    Visualize original image, ground truth, and predicted mask
    for the slices in `slice_indices`.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    slice_names = ['Upper Third', 'Middle', 'Lower Third']
    plt.figure(figsize=(20, 20))
    plt.suptitle("Inference Results (Axial View)", fontsize=16)

    for idx, (slice_index, slice_name) in enumerate(zip(slice_indices, slice_names)):
        # Extract & transform the slice
        vol_slice = transform_slice(get_slice(volume, slice_index, "Axial"), "Axial")
        mask_slice = transform_slice(get_slice(mask, slice_index, "Axial"), "Axial")

        # Normalize the volume slice to [0, 255] for plotting
        if vol_slice.max() != vol_slice.min():
            vol_slice = (vol_slice - vol_slice.min()) / (vol_slice.max() - vol_slice.min())
        vol_slice = (vol_slice * 255).astype(np.uint8)

        # Convert mask to [0, 255]
        mask_slice = (mask_slice * 255).astype(np.uint8)

        # Create PyTorch tensors
        vol_torch = torch.tensor(vol_slice, dtype=torch.uint8).unsqueeze(0)  # (1, H, W)
        mask_torch = torch.tensor(mask_slice, dtype=torch.uint8).unsqueeze(0)  # (1, H, W)

        # Expand volume to 3 channels
        vol_torch = vol_torch.repeat(3, 1, 1)  # (3, H, W)

        # Binarize the mask for ground truth
        mask_torch = (mask_torch > 0.5).float()

        # Compute bounding box from mask
        y_indices, x_indices = torch.where(mask_torch[0] > 0)
        if len(x_indices) > 0 and len(y_indices) > 0:
            x_min, x_max = x_indices.min().item(), x_indices.max().item()
            y_min, y_max = y_indices.min().item(), y_indices.max().item()
            bbox = np.array([[x_min, y_min, x_max, y_max]])
        else:
            # If there's no nonzero region, skip
            continue

        # Forward pass through the model
        model_input = vol_torch.unsqueeze(0).float().to(device)
        with torch.no_grad():
            predicted_mask_logits = model(model_input, bbox).squeeze().sigmoid().cpu()
        
        predicted_mask_bin = (predicted_mask_logits > 0.5).float()

        # Convert to numpy for plotting
        vol_plot = vol_torch[0].numpy()
        gt_plot = mask_torch[0].numpy()
        pred_plot = predicted_mask_bin.numpy()

        row = idx + 1
        plt.subplot(3, 4, row * 4 - 3)
        plt.title(f"{slice_name} Slice - Original Image")
        plt.imshow(vol_plot, cmap="gray")
        plt.axis('off')

        plt.subplot(3, 4, row * 4 - 2)
        plt.title(f"{slice_name} Slice - Ground Truth")
        plt.imshow(gt_plot, cmap="gray")
        plt.axis('off')

        plt.subplot(3, 4, row * 4 - 1)
        plt.title(f"{slice_name} Slice - Predicted Mask")
        plt.imshow(pred_plot, cmap="gray")
        plt.axis('off')

        plt.subplot(3, 4, row * 4)
        plt.title(f"{slice_name} Slice - Overlay")
        plt.imshow(vol_plot, cmap="gray")
        plt.imshow(pred_plot, alpha=0.5, cmap="jet")
        plt.axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"inference_example_{current_date}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")


def run_inference(
    checkpoint_path,
    model_type,
    data_dir,
    output_dir,
    visualize=True
):
    """
    Load a trained MedSAM model from checkpoint, read volumes & masks,
    and run inference on a sample volume for demonstration.
    
    Args:
        checkpoint_path (str): Path to the model weights (e.g., 'best_model.pth').
        model_type (str): Key for segment_anything registry (e.g., 'vit_b').
        data_dir (str): Directory containing subfolders with volumes and masks.
        output_dir (str): Where to save visualization images.
        visualize (bool): Whether to save inference visualization.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the base SAM model & initialize MedSAM
    print(f"Loading SAM model from registry: {model_type}")
    base_sam_model = sam_model_registry[model_type](checkpoint=None)  # We'll load custom weights next
    med_sam_model = MedSAM(base_sam_model).to(device)

    # 2. Load the checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    med_sam_model.load_state_dict(checkpoint)

    med_sam_model.eval()  # Set to inference mode

    # 3. Read some data (volumes + masks)
    print(f"Reading NRRD data from {data_dir} ...")
    volumes = read_nrrd_files(data_dir, 'volume')
    masks = read_nrrd_files(data_dir, 'mask')
    if len(volumes) == 0 or len(masks) == 0:
        print("No valid data found for inference. Exiting.")
        return

    # For demonstration, pick the last volume/mask pair
    volume = volumes[-1]
    mask = masks[-1]

    # 4. Identify slices of interest
    start_slice, end_slice = find_nonzero_range(mask)
    if start_slice is None or end_slice is None:
        print("No non-zero slices found in mask. Exiting.")
        return

    # Get representative slices (upper, middle, lower)
    slice_indices = get_representative_slices(start_slice, end_slice)
    print(f"Selected slice indices for visualization: {slice_indices}")

    # 5. Visualization (optional)
    if visualize:
        visualize_inference_results(
            model=med_sam_model,
            volume=volume,
            mask=mask,
            slice_indices=slice_indices,
            device=device,
            output_dir=output_dir
        )
    else:
        print("Visualization disabled. Inference complete.")


def main():
    """
    Example command-line interface for running inference.
    """
    parser = argparse.ArgumentParser(description="MedSAM Inference Script")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint file (e.g., 'best_model.pth')."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vit_b",
        help="SAM model type from segment_anything registry (e.g., 'vit_b', 'vit_l', 'vit_h')."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the dataset directory containing 'volume' and 'mask' subfolders."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_outputs",
        help="Directory to save inference visualization images."
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="If provided, inference results won't be visualized or saved."
    )

    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint_path,
        model_type=args.model_type,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        visualize=not args.no_vis
    )

if __name__ == "__main__":
    main()

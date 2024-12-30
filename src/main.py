# main.py
import os
import argparse

from src.train import run_tuning

def main():
    # Example: parse any custom command line arguments
    parser = argparse.ArgumentParser(description="MedSAM Training with Ray Tune")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/path/to/your/dataset",
        help="Path to the dataset directory containing subfolders with volumes and masks."
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default="/path/to/sam_vit_b_checkpoint.pth",
        help="Path to the pretrained SAM model checkpoint."
    )
    parser.add_argument(
        "--ray_logs",
        type=str,
        default="/path/to/raytune_logs",
        help="Path where Ray Tune logs and checkpoints are stored."
    )
    args = parser.parse_args()

    # Set environment variables or pass them in another way
    os.environ["DATA_DIRECTORY"] = args.data_dir
    
    # If needed, you might also modify the run_tuning config directly. 
    # But as shown, you can keep it inside train.py or pass arguments:
    # For example, you could override the default Ray Tune config:
    #   config = {
    #       "model_type": "vit_b",
    #       "checkpoint": args.sam_checkpoint,
    #       ...
    #   }
    # Then pass it to a modified run_tuning(config).

    # For now, we'll just call run_tuning directly, 
    # which uses the default config in train.py
    print(f"Using dataset directory: {args.data_dir}")
    print(f"Using SAM checkpoint: {args.sam_checkpoint}")
    print(f"Ray logs stored at: {args.ray_logs}")

    run_tuning()

if __name__ == "__main__":
    main()

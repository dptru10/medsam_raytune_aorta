# MedSAM Aorta Segmentation with Ray Tune

This repository provides code for training and evaluating a **Segment Anything Model (SAM)**-based network, called **MedSAM**, for aorta segmentation in 3D medical volumes. The project leverages **Ray Tune** for hyperparameter tuning and **MONAI** for medical-specific loss functions and metrics. 

## Directory Structure

```
.
└── src
    ├── dataset.py
    ├── inference.py
    ├── main.py
    ├── model.py
    ├── reference.py
    ├── train.py
├── requirements.txt
├── README.md
```

- **`dataset.py`**: Functions and classes related to reading **.nrrd** files, slicing volumes, and creating the PyTorch `Dataset`.  
- **`model.py`**: The custom `MedSAM` model class that uses the SAM architecture.  
- **`train.py`**: Training logic (the function passed to Ray Tune) and a helper function `run_tuning()` to configure and launch the Ray Tune hyperparameter search.  
- **`main.py`**: A high-level entry point that calls `run_tuning()`.  
- **`inference.py`**: Script for loading a trained model checkpoint, running inference on a sample volume, and optionally visualizing the results.  
- **`reference.py`**: (Optional) A file to hold extra code snippets or references.  
- **`requirements.txt`**: Python dependencies required for this project.  

## Data Layout

Your dataset directory should have a structure like:

```
DATA_DIRECTORY/
├── Patient001/
│   ├── volume/
│   │   └── volume.nrrd
│   └── mask/
│       └── mask.nrrd
├── Patient002/
│   ├── volume/
│   │   └── volume.nrrd
│   └── mask/
│       └── mask.nrrd
...
```

Each patient folder contains a `volume/` and a `mask/` subdirectory, each with a single `.nrrd` file. Adjust names to match your actual data.

## Installation

1. **Create a virtual environment** (optional but recommended):

   ```bash
   conda create -n medsam_env python=3.9 -y
   conda activate medsam_env
   ```
   or using `venv`:
   ```bash
   python3 -m venv medsam_env
   source medsam_env/bin/activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
   
   - Make sure your **PyTorch** version is compatible with your CUDA version, if you plan on using a GPU.  
   - You may need to install **segment-anything** from GitHub if it is not available on PyPI:
     ```bash
     pip install git+https://github.com/facebookresearch/segment-anything.git
     ```

3. **Download or locate the SAM checkpoint** (e.g. `sam_vit_b_01ec64.pth` or your custom checkpoint). Keep note of the path for the training script.

## Usage

### 1. Training & Hyperparameter Tuning

1. **Set your dataset directory**:  
   Either set the environment variable or pass it as an argument:
   ```bash
   export DATA_DIRECTORY=/absolute/path/to/your/data
   ```
   
2. **Run the main training script**:
   ```bash
   cd src
   python main.py \
       --data_dir /absolute/path/to/your/data \
       --sam_checkpoint /absolute/path/to/sam_vit_b_checkpoint.pth \
       --ray_logs /absolute/path/to/raytune_logs
   ```
   - `--data_dir` sets the data directory (also sets an env var).  
   - `--sam_checkpoint` is the path to your SAM checkpoint file.  
   - `--ray_logs` sets the directory where Ray Tune logs and checkpoints are stored.  

Ray Tune will run multiple trials, each with a different hyperparameter configuration, and will log metrics like `train_loss`, `train_dice`, `val_loss`, and `val_dice`. The best model (highest `val_dice`) is checkpointed automatically.

### 2. Inference & Visualization

After training, you will have a best checkpoint (`checkpoint.pt`) or a saved `best_model.pth`. You can use **`inference.py`** to run inference on a chosen volume:

```bash
cd src
python inference.py \
    --checkpoint_path /absolute/path/to/best_model.pth \
    --model_type vit_b \
    --data_dir /absolute/path/to/your/data \
    --output_dir ./inference_outputs
```

- **`--checkpoint_path`**: The path to your trained model file.  
- **`--model_type`**: One of `vit_b`, `vit_l`, or `vit_h`, as per your SAM variant.  
- **`--data_dir`**: Same directory layout as used for training.  
- **`--output_dir`**: Where to save overlay images (`inference_example_YYYYMMDD_HHMMSS.png`).  
- **`--no_vis`**: If provided, will skip visualization and just run inference.

## Notes & Tips

- **Check GPU usage**: If you have multiple GPUs, `train.py` automatically wraps the model in `nn.DataParallel` when `torch.cuda.device_count() > 1`. For better performance on multi-GPU setups, consider `DistributedDataParallel`.
- **Adjust batch size**: The batch size is tuned by Ray Tune (`[2, 4, 8, 16]`). You may need to adjust these ranges if you run out of GPU memory or want a larger batch size.
- **Memory considerations**: The current `AortaDataset2D` class loads all slices from each volume into memory. If you have very large volumes or many patients, you may want to refactor for on-the-fly slicing.
- **Loss function**: The code uses **`DiceCELoss`** from MONAI plus an additional **binary cross-entropy** term. If you see over-penalizing of the CE component, you might remove or weight one of these terms to taste.
- **Check the bounding box logic**: The bounding box is derived from the nonzero region of the mask. Ensure that this is appropriate for your dataset and your segmentation task.

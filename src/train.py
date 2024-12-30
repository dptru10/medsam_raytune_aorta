# train.py

import os
import tempfile
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.train import Checkpoint

from segment_anything import sam_model_registry

# Local imports
from src.dataset import (
    read_nrrd_files, AortaDataset2D
)
from src.model import MedSAM


def train_model(config):
    """
    Ray Tune-compatible training function.
    This is called for each hyperparameter trial.
    """
    # ------------------------------------------------------------------
    # Load volumes and masks
    # ------------------------------------------------------------------
    data_dir = os.environ.get("DATA_DIRECTORY", "./data")
    volumes = read_nrrd_files(data_dir, 'volume')
    masks = read_nrrd_files(data_dir, 'mask')

    if not volumes or not masks:
        print("Error: No valid data found. Exiting training.")
        return

    # ------------------------------------------------------------------
    # Train / Val split & dataset creation
    # ------------------------------------------------------------------
    train_volumes, val_volumes, train_masks, val_masks = train_test_split(
        volumes, masks, test_size=0.2, random_state=42
    )

    train_dataset = AortaDataset2D(train_volumes, train_masks)
    val_dataset = AortaDataset2D(val_volumes, val_masks)

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

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Retrieve the SAM model from the registry
    sam_model = sam_model_registry[config["model_type"]](
        checkpoint=config["checkpoint"]
    )
    model = MedSAM(sam_model).to(device)

    # Parallelize if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # ------------------------------------------------------------------
    # Optimizer & Loss
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    dice_ce_loss = DiceCELoss(to_onehot_y=False, sigmoid=True)

    # MONAI's DiceMetric
    best_dice = 0.0

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        step = 0
        dice_metric = DiceMetric(include_background=False, reduction="mean")

        for batch_data in train_loader:
            step += 1
            inputs = batch_data['image'].to(device).float()  
            labels = batch_data['label'].unsqueeze(1).to(device).float()  
            bboxes = batch_data['bbox'].to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(inputs, bboxes)
            
            # Combine Dice + BCE 
            loss = dice_ce_loss(outputs, labels) + F.binary_cross_entropy_with_logits(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = (outputs > 0.5).float()
            dice_metric(y_pred=pred, y=labels)

        epoch_loss /= step
        train_dice = dice_metric.aggregate().item()
        dice_metric.reset()

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_step = 0
        val_dice_metric = DiceMetric(include_background=False, reduction="mean")

        with torch.no_grad():
            for val_data in val_loader:
                val_step += 1
                val_inputs = val_data['image'].to(device).float()
                val_labels = val_data['label'].unsqueeze(1).to(device).float()
                val_bboxes = val_data['bbox'].to(device, dtype=torch.float32)

                val_outputs = model(val_inputs, val_bboxes)
                v_loss = dice_ce_loss(val_outputs, val_labels) + \
                         F.binary_cross_entropy_with_logits(val_outputs, val_labels)
                
                val_loss += v_loss.item()

                val_pred = (val_outputs > 0.5).float()
                val_dice_metric(y_pred=val_pred, y=val_labels)

        val_loss /= val_step
        val_dice = val_dice_metric.aggregate().item()
        val_dice_metric.reset()

        # --------------------------------------------------------------
        # Report metrics to Ray Tune
        # --------------------------------------------------------------
        metrics = {
            "train_loss": epoch_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_dice": val_dice
        }

        # If we have a new best dice, save checkpoint
        if val_dice > best_dice:
            best_dice = val_dice
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
                torch.save(model.state_dict(), checkpoint_path)
                session.report(metrics, checkpoint=Checkpoint.from_directory(tmpdir))
        else:
            session.report(metrics)


def run_tuning():
    """
    Define hyperparameters, scheduler, reporter, and run Ray Tune.
    This function can be called from main.py to start the tuning process.
    """
    # Example hyperparameter config
    config = {
        "model_type": "vit_b",
        "checkpoint": "/path/to/sam_vit_b_checkpoint.pth",
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "num_epochs": 10
    }

    # ASHAScheduler for early stopping
    scheduler = ASHAScheduler(
        metric="val_dice",
        mode="max",
        max_t=10,
        grace_period=2,
        reduction_factor=2
    )

    # CLIReporter for better console logs
    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size", "weight_decay"],
        metric_columns=["train_loss", "val_loss", "train_dice", "val_dice", "training_iteration"]
    )

    # Directory to store Ray Tune logs/checkpoints
    storage_path = "/path/to/raytune_logs"
    os.makedirs(storage_path, exist_ok=True)

    result = tune.run(
        train_model,
        resources_per_trial={"cpu": 4, "gpu": 1 if torch.cuda.is_available() else 0},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter,
        storage_path=storage_path,
        log_to_file=True
    )

    # Retrieve best trial
    best_trial = result.get_best_trial("val_dice", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {:.4f}".format(
        best_trial.last_result["val_loss"]))
    print("Best trial final validation dice: {:.4f}".format(
        best_trial.last_result["val_dice"]))

    # Example of loading best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_model = sam_model_registry[best_trial.config["model_type"]](
        checkpoint=best_trial.config["checkpoint"]
    )
    best_model = MedSAM(sam_model).to(device)

    best_checkpoint = best_trial.checkpoint
    with best_checkpoint.as_directory() as checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        best_model.load_state_dict(torch.load(checkpoint_path))
    
    print("Best model loaded. You can now run further evaluation or inference.")

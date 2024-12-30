# model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MedSAM(nn.Module):
    """
    A custom wrapper around the SAM model for medical segmentation tasks.
    It reuses SAM's image_encoder, prompt_encoder, and mask_decoder.
    The prompt encoder is frozen to avoid backprop through prompt embeddings.

    Usage Example (in another script):
        from segment_anything import sam_model_registry
        from model import MedSAM

        sam_base = sam_model_registry["vit_b"](checkpoint="path/to/checkpoint.pth")
        med_sam_model = MedSAM(sam_base).to(device)
    """
    def __init__(self, sam_model):
        super().__init__()
        # Extract SAM submodules
        self.image_encoder = sam_model.image_encoder
        self.mask_decoder = sam_model.mask_decoder
        self.prompt_encoder = sam_model.prompt_encoder

        # Freeze the prompt encoder so its gradients are not updated
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        """
        Forward pass for MedSAM.

        Args:
            image (torch.Tensor): Image tensor of shape (B, 3, H, W).
            box (torch.Tensor or np.ndarray): Bounding box coordinates, 
                typically shaped (B, 1, 4) or (B, 4).

        Returns:
            torch.Tensor: Segmentation mask logits of shape (B, 1, H, W).
        """
        # Encode image
        image_embedding = self.image_encoder(image)
        
        # Prompt encoder is used in inference mode (no gradient)
        with torch.no_grad():
            if isinstance(box, np.ndarray):
                box = torch.from_numpy(box)
            
            # Ensure box is of shape (B, 1, 4)
            if len(box.shape) == 2:
                box = box.unsqueeze(1)
            
            box_torch = box.to(image.device, dtype=torch.float32)

            # Get sparse and dense prompt embeddings
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )

        # Decode segmentation mask
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # Upsample masks to original image resolution
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        
        return ori_res_masks

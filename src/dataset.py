import os
import nrrd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def load_nrrd_file(file_path):
    """
    Load an NRRD file and return the data and header.
    """
    data, header = nrrd.read(file_path)
    return data, header

def get_representative_slices(start, end):
    """
    Given a start and end slice index, return a list of up to three
    representative slices: the first, middle, and last slice.
    """
    if start == end:
        return [start, start, start]
    total_slices = end - start + 1
    return [
        start,
        start + total_slices // 2,
        end
    ]

def get_slice(data, slice_index, view="Axial"):
    """
    Extract a 2D slice from 3D data based on the specified view.
    
    Args:
        data (np.ndarray): The 3D volume (shape: [Z, Y, X] or similar).
        slice_index (int): Which slice to extract along the chosen axis.
        view (str): One of 'Axial', 'Coronal', 'Sagittal'.
    
    Returns:
        np.ndarray: The extracted 2D slice.
    """
    if view == "Axial":
        return data[:, :, slice_index]
    elif view == "Coronal":
        return data[:, slice_index, :]
    elif view == "Sagittal":
        return data[slice_index, :, :]
    else:
        raise ValueError(f"Unsupported view: {view}")

def transform_slice(slice_image, view="Axial"):
    """
    Apply transformations (rotations/flips) to a 2D slice for consistent orientation.
    - Rotate the image 90 degrees clockwise
    - Flip left-right
    - Flip top-bottom for Coronal and Sagittal
    
    Args:
        slice_image (np.ndarray): The 2D slice.
        view (str): One of 'Axial', 'Coronal', 'Sagittal'.
    
    Returns:
        np.ndarray: Transformed 2D slice.
    """
    # Rotate the image 90 degrees clockwise
    slice_image = np.rot90(slice_image, k=-1)
    # Flip the image from left to right
    slice_image = np.fliplr(slice_image)
    # Flip top to bottom for coronal/sagittal
    if view in ["Coronal", "Sagittal"]:
        slice_image = np.flipud(slice_image)
    return slice_image

def find_nonzero_range(mask_data):
    """
    Find the first and last slice indices in a 3D mask that contain nonzero values.
    
    Args:
        mask_data (np.ndarray): 3D mask array.
    
    Returns:
        tuple: (start_slice, end_slice) or (None, None) if no non-zero slices.
    """
    nonzero_slices = np.where(np.any(mask_data, axis=(0, 1)))[0]
    if len(nonzero_slices) == 0:
        return None, None
    return nonzero_slices[0], nonzero_slices[-1]

def read_nrrd_files(data_path, data_type):
    """
    Read NRRD files from each folder in `data_path`.
    
    Args:
        data_path (str): Directory containing subfolders with 'volume' or 'mask' data.
        data_type (str): 'volume' or 'mask', determining which subfolder to read from.
    
    Returns:
        list: A list of NRRD data arrays.
    """
    folders = [
        folder for folder in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, folder))
    ]
    data_list = []

    for folder in folders:
        try:
            folder_path = os.path.join(data_path, folder)
            nrrd_path = os.path.join(
                folder_path, 
                'volume' if data_type == 'volume' else 'mask'
            )
            nrrd_files = [file for file in os.listdir(nrrd_path) if file.endswith('.nrrd')]
            print(f"Reading {data_type} from {folder}")

            if len(nrrd_files) != 1:
                print(
                    f"Skipping {folder}: Expected 1 NRRD file, "
                    f"found {len(nrrd_files)} in {nrrd_path}"
                )
                continue

            file_path = os.path.join(nrrd_path, nrrd_files[0])
            data, _ = load_nrrd_file(file_path)
            data_list.append(data)

        except FileNotFoundError as e:
            print(f"Skipping {folder}: {str(e)}")
        except Exception as e:
            print(f"Error processing {folder}: {str(e)}")

    return data_list

class AortaDataset2D(Dataset):
    """
    A PyTorch Dataset for loading 2D axial slices from 3D volumes and corresponding masks.
    Each volume is sliced in the axial plane from the first non-zero mask slice
    to the last non-zero mask slice.

    Transformation logic:
    - Convert each slice to PIL and resize to (1024, 1024)
    - Convert back to tensor
    - Binarize mask
    - Compute bounding box from the mask
    """
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
            transforms.Resize(
                (1024, 1024), 
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])
        
        # Build dataset in memory
        for volume, mask in zip(volumes, masks):
            start_slice, end_slice = find_nonzero_range(mask)
            if start_slice is None or end_slice is None:
                continue

            # Iterate over each slice in the range
            for i in range(start_slice, end_slice + 1):
                volume_slice = transform_slice(get_slice(volume, i, "Axial"), "Axial")
                mask_slice = transform_slice(get_slice(mask, i, "Axial"), "Axial")
                
                # Normalize volume slice to [0, 1]
                if volume_slice.max() != volume_slice.min():
                    volume_slice = (
                        (volume_slice - volume_slice.min()) /
                        (volume_slice.max() - volume_slice.min())
                    )
                
                # Convert to uint8 for PIL Image
                volume_slice = (volume_slice * 255).astype(np.uint8)
                mask_slice = (mask_slice * 255).astype(np.uint8)
                
                # Apply transforms -> Tensors
                volume_slice = self.image_transform(volume_slice)  # (1, 1024, 1024)
                mask_slice = self.mask_transform(mask_slice)       # (1, 1024, 1024)
                
                # Expand volume from 1-channel to 3-channel
                volume_slice = volume_slice.repeat(3, 1, 1)  # (3, 1024, 1024)
                
                # Binarize the mask
                mask_slice = (mask_slice > 0.5).float()  # Still shape (1, 1024, 1024)
                
                # Compute bounding box
                y_indices, x_indices = torch.where(mask_slice[0] > 0)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    x_min, x_max = x_indices.min().item(), x_indices.max().item()
                    y_min, y_max = y_indices.min().item(), y_indices.max().item()
                    bbox = np.array([x_min, y_min, x_max, y_max])
                else:
                    # No nonzero in this slice (shouldn't happen given find_nonzero_range, but just in case)
                    continue
                
                self.data.append({
                    'image': volume_slice,   # (3, 1024, 1024)
                    'label': mask_slice[0],  # (1024, 1024)
                    'bbox': bbox             # (4,) numpy array
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

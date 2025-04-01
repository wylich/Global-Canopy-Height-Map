import os
import zipfile
import random
from shutil import copyfile
from pathlib import Path

def extract_and_split_samples(
    zip_path: str = "data/samples.zip",
    output_dir: str = "data/samples1",
    n_samples: int = 1000,
    val_ratio: float = 0.2
):
    """
    Extract samples from zip and split into train/val sets
    
    Args:
        zip_path: Path to samples.zip
        output_dir: Output directory (will contain train/val subfolders)
        n_samples: Total number of samples to extract
        val_ratio: Ratio of samples for validation (0.2 = 20%)
    """
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Extract and split samples
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        all_files = [f for f in zip_ref.namelist() if f.endswith(('.npz', '.png', '.tif'))]  # Adjust extensions
        
        # Randomly select samples
        selected_files = random.sample(all_files, min(n_samples, len(all_files)))
        random.shuffle(selected_files)
        
        # Split into train/val
        split_idx = int(len(selected_files) * (1 - val_ratio))
        train_files = selected_files[:split_idx]
        val_files = selected_files[split_idx:]
        
        # Extract files
        for i, file in enumerate(train_files):
            zip_ref.extract(file, train_dir)
            print(f"\rExtracted {i+1}/{len(train_files)} train samples", end="")
        
        for i, file in enumerate(val_files):
            zip_ref.extract(file, val_dir)
            print(f"\rExtracted {i+1}/{len(val_files)} val samples", end="")
    
    print(f"\nDone! Extracted {len(train_files)} train and {len(val_files)} val samples to {output_dir}")

if __name__ == "__main__":
    extract_and_split_samples()
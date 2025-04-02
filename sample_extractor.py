import pandas as pd
import numpy as np
import os
from pathlib import Path

def create_train_val_splits(csv_path, n_samples=None, val_ratio=0.2, random_state=42):
    """
    Creates train/validation splits from a samples CSV file.
    
    Args:
        csv_path (str): Path to input CSV file (e.g., 'datasets_pytorch/ai4forest_camera/samples.csv')
        n_samples (int): Total number of samples to use (None = use all)
        val_ratio (float): Ratio for validation set (0.0-1.0)
        random_state (int): Random seed for reproducibility
    """
    # Read the original CSV
    df = pd.read_csv(csv_path)
    
    # Subsample if requested
    if n_samples is not None:
        df = df.sample(n=min(n_samples, len(df)), random_state=random_state)
    
    # Split into train/val
    val_size = int(len(df) * val_ratio)
    train_df = df.iloc[:-val_size]
    val_df = df.iloc[-val_size:]
    
    # Create output paths
    base_dir = Path(csv_path).parent
    train_path = base_dir / "train.csv"
    val_path = base_dir / "val.csv"
    
    # Save new CSVs
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"Created splits:\n"
          f"- Train: {len(train_df)} samples -> {train_path}\n"
          f"- Val: {len(val_df)} samples -> {val_path}")

# Example usage:
if __name__ == "__main__":
    csv_path = "datasets_pytorch/ai4forest_camera/samples.csv"
    create_train_val_splits(
        csv_path=csv_path,
        n_samples=10000,  # Set to None to use all samples
        val_ratio=0.2,    # 20% validation
        random_state=42   # For reproducibility
    )
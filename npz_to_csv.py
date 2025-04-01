# we convert all .npz files in the data folder to one csv file

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# tried out for validation set first
def npz_to_csv(input_dir="data/samples1/val/samples", output_csv="data/samples1/val/val.csv"):
    """
    Convert NPZ files to a CSV file with flattened arrays
    
    Args:
        input_dir: Directory containing NPZ files
        output_csv: Output CSV file path
    """
    # Find all NPZ files
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    print(f"Found {len(npz_files)} files.")
    
     # Prepare DataFrame storage
    data_rows = []
    
    for npz_file in tqdm(npz_files, desc="Processing NPZ files"):
        file_path = os.path.join(input_dir, npz_file)
        
        try:
            # Load NPZ file
            with np.load(file_path) as data:
                # Handle different NPZ structures
                if 'data' in data:
                    image = data['data']  # Assuming shape (14, 512, 512)
                    # Flatten the image array
                    image_flat = image.flatten()
                    # Convert to string representation for CSV
                    image_str = ' '.join(map(str, image_flat))
                    
                    # Handle labels if they exist
                    labels_str = ''
                    if 'labels' in data:
                        labels = data['labels']
                        labels_flat = labels.flatten()
                        labels_str = ' '.join(map(str, labels_flat))
                    
                    data_rows.append({
                        'filename': npz_file,
                        'image': image_str,
                        'labels': labels_str
                    })
        except Exception as e:
            print(f"\nError processing {npz_file}: {str(e)}")
            continue
    
    # Create DataFrame and save to CSV
    if data_rows:
        df = pd.DataFrame(data_rows)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nSuccessfully saved {len(df)} samples to {output_csv}")
    else:
        print("No valid data processed.")

if __name__ == "__main__":
    npz_to_csv()
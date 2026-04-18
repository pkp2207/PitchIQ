import os
import zipfile
import subprocess
from pathlib import Path

import sys

def download_kaggle_dataset(dataset_name, download_path):
    print(f"Downloading {dataset_name} to {download_path}...")
    try:
        subprocess.run([
            sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset_name, "-p", str(download_path), "--unzip"
        ], check=True)
        print(f"Successfully downloaded and extracted {dataset_name}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {dataset_name}. Make sure kaggle API is configured.")
        print(e)
    except FileNotFoundError:
        print("Kaggle CLI not found. Please install it using 'pip install kaggle' and configure your API key.")

if __name__ == "__main__":
    # Define project root (two levels up from this script)
    project_root = Path(__file__).resolve().parent.parent
    data_raw_dir = project_root / "data" / "raw"
    
    # Create directory if it doesn't exist
    os.makedirs(data_raw_dir, exist_ok=True)
    
    datasets = [
        "martj42/international-football-results-from-1872-to-2017",
        "davidcariboo/player-scores"
    ]
    
    for dataset in datasets:
        download_kaggle_dataset(dataset, data_raw_dir)

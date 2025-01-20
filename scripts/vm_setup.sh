#!/bin/bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y git python3-pip python3-opencv python3.10-venv
git config --global user.email luca.zanolo@outlook.it
git config --global user.name Luca

python3 -m venv .venv
source .venv/bin/activate

echo "Installing necessary Python packages..."
pip install -r scripts/vm_requirements.txt

# Check if the dataset exists, otherwise download it
DATASET_DIR="./dataset"
REFCOCOG_DIR="$DATASET_DIR/refcocog"
DATASET_FILE="$DATASET_DIR/refcocog.tar.gz"
DATASET_GOOGLE_DRIVE_URL="https://drive.google.com/uc?id=1xijq32XfEm6FPhUb7RsZYWHc2UuwVkiq"

# Ensure the dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Creating dataset folder..."
    mkdir -p $DATASET_DIR
fi

# Check if the RefCOCOg dataset already exists as a folder
if [ ! -d "$REFCOCOG_DIR" ]; then
    # If not, check if the .tar.gz file is present, otherwise download it
    if [ ! -f "$DATASET_FILE" ]; then
        echo "Downloading RefCOCOg dataset..."
        # Install gdown if necessary for downloading from Google Drive
        pip install gdown
        gdown $DATASET_GOOGLE_DRIVE_URL -O "$DATASET_FILE"
    fi

    echo "Extracting dataset..."
    tar -xzvf "$DATASET_FILE" -C "$DATASET_DIR"
    
    echo "Cleaning up..."
    rm "$DATASET_FILE"  # Remove the downloaded tar.gz file after extraction
else
    echo "RefCOCOg dataset already exists. Skipping download."
fi

echo "Setup complete."
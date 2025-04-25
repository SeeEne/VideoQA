# Filename: extract_embeddings.py
# Purpose: Extracts and saves embeddings from a vision model for specified dataset splits.

import torch
import numpy as np
from PIL import Image
import os
import sys
import yaml
import argparse
from tqdm.notebook import tqdm # Or from tqdm import tqdm

# Import helpers from data_preprocess
try:
    from data_preprocess import get_image_dataloader, extract_embedding
except ImportError as e:
    print(f"Error importing from data_preprocess.py: {e}")
    sys.exit(1)

# Import model/processor loaders
try:
     from transformers import AutoProcessor, AutoModel
except ImportError:
     print("Error: `transformers` library not installed. Please install it.")
     sys.exit(1)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Extract and save vision model embeddings.")
parser.add_argument('--config_file', type=str, default='config.yaml',
                    help='Path to the YAML configuration file')
# Allow overriding specific parameters if needed, otherwise use config
parser.add_argument('--vision_model_id', type=str, default=None, help='Override VISION_MODEL_ID from config')
parser.add_argument('--dataset_path', type=str, default=None, help='Override DATASET_PATH from config')
parser.add_argument('--dataset_name', type=str, default=None, help='Override PROBING_DATASET_NAME from config')
parser.add_argument('--embedding_path', type=str, default=None, help='Override EMBEDDING_BASE_PATH from config')
parser.add_argument('--image_size', type=int, default=None, help='Override IMAGE_SIZE from config')
parser.add_argument('--batch_size', type=int, default=None, help='Override BATCH_SIZE for extraction')

args = parser.parse_args()

# --- Load Configuration ---
config = {}
probe_config = {}
try:
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    config_path = os.path.join(script_dir, args.config_file)
    if not os.path.exists(config_path): raise FileNotFoundError(f"Config not found: {config_path}")
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f: config = yaml.safe_load(f) or {}
    probe_config = config.get('ProbeTraining', {}) # Get the relevant section

    # --- Determine Effective Parameters (Args override Config) ---
    VISION_MODEL_ID = args.vision_model_id or probe_config.get('VISION_MODEL_ID')
    DATASET_PATH = args.dataset_path or probe_config.get('DATASET_PATH')
    PROBING_DATASET_NAME = args.dataset_name or probe_config.get('PROBING_DATASET_NAME')
    EMBEDDING_BASE_PATH = args.embedding_path or probe_config.get('EMBEDDING_BASE_PATH')
    IMAGE_SIZE = args.image_size or int(probe_config.get('IMAGE_SIZE', 224))
    BATCH_SIZE = args.batch_size or int(probe_config.get('BATCH_SIZE', 64)) # Use probe batch size or specific extraction batch size
    NUM_WORKERS = int(probe_config.get('NUM_WORKERS', 0)) # Use probe workers for data loading

    # Validate essential params
    if not all([PROBING_DATASET_NAME, VISION_MODEL_ID, EMBEDDING_BASE_PATH, DATASET_PATH]):
         raise ValueError("Missing essential config/args: DATASET_NAME, VISION_MODEL_ID, EMBEDDING_BASE_PATH, DATASET_PATH")

except Exception as e:
    print(f"Error loading/parsing configuration: {e}")
    sys.exit(1)


# --- Main Extraction Logic ---
if __name__ == "__main__":
    print("\n--- Embedding Extraction ---")
    print(f" Vision Model: {VISION_MODEL_ID}")
    print(f" Dataset:      {PROBING_DATASET_NAME} ({DATASET_PATH})")
    print(f" Output Path:  {EMBEDDING_BASE_PATH}")
    print(f" Image Size:   {IMAGE_SIZE}")
    print(f" Batch Size:   {BATCH_SIZE}")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory structure
    safe_model_id_name = VISION_MODEL_ID.replace('/', '_').replace('\\', '_')
    output_model_dir = os.path.join(EMBEDDING_BASE_PATH, safe_model_id_name)
    os.makedirs(output_model_dir, exist_ok=True)
    print(f"Embeddings will be saved in: {output_model_dir}")

    # --- Load Vision Model ---
    print("Loading vision model and processor...")
    try:
        processor = AutoProcessor.from_pretrained(VISION_MODEL_ID)
        model = AutoModel.from_pretrained(VISION_MODEL_ID).to(device)
        model.eval()
        for param in model.parameters(): param.requires_grad = False # Ensure frozen
        print(f"Loaded vision model: {type(model).__name__}")
    except Exception as e:
        print(f"Error loading vision model: {e}"); sys.exit(1)

    # --- Process Splits (train, val, potentially test) ---
    # Assumes dataset has 'train' and 'val' subdirectories
    for split in ['train', 'val']: # Add 'test' if you have it
        print(f"\nProcessing split: '{split}'...")
        split_data_path = os.path.join(DATASET_PATH, split)
        if not os.path.isdir(split_data_path):
            print(f"Warning: Directory for split '{split}' not found at '{split_data_path}'. Skipping.")
            continue

        # 1. Create Image DataLoader for this split
        # Use is_train=False for transforms to avoid augmentation during extraction
        dataloader, num_classes = get_image_dataloader(
            data_path=split_data_path,
            batch_size=BATCH_SIZE,
            image_size=IMAGE_SIZE,
            is_train=False, # No augmentation for feature extraction
            num_workers=NUM_WORKERS
        )

        if dataloader is None:
            print(f"Failed to create DataLoader for split '{split}'. Skipping.")
            continue

        # 2. Iterate and Extract Embeddings
        all_embeddings = []
        all_labels = []
        print(f"Extracting embeddings for {len(dataloader.dataset)} images in '{split}' split...")
        for image_batch, label_batch in tqdm(dataloader, desc=f"Extracting {split}"):
            # extract_embedding expects tensor batch, model, processor, device
            embeddings = extract_embedding(image_batch, model, processor, device)

            if embeddings is not None:
                all_embeddings.append(embeddings) # Keep embeddings on CPU
                all_labels.append(label_batch.clone()) # Store corresponding labels
            else:
                print(f"Warning: Failed to extract embedding for a batch in split '{split}'.")

        # 3. Concatenate and Save
        if not all_embeddings:
             print(f"Error: No embeddings extracted for split '{split}'. Cannot save.")
             continue

        try:
            final_embeddings = torch.cat(all_embeddings, dim=0)
            final_labels = torch.cat(all_labels, dim=0)
            print(f"Final {split} embeddings shape: {final_embeddings.shape}")
            print(f"Final {split} labels shape: {final_labels.shape}")

            # Define output file paths
            embed_file = os.path.join(output_model_dir, f"{PROBING_DATASET_NAME}_{split}_embeddings.pt")
            label_file = os.path.join(output_model_dir, f"{PROBING_DATASET_NAME}_{split}_labels.pt")

            # Save the tensors
            torch.save(final_embeddings, embed_file)
            torch.save(final_labels, label_file)
            print(f"Saved {split} embeddings to: {embed_file}")
            print(f"Saved {split} labels to: {label_file}")

        except Exception as e:
            print(f"Error concatenating or saving embeddings/labels for split '{split}': {e}")

    print("\n--- Embedding Extraction Complete ---")
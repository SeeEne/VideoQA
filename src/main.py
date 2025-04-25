# Filename: main.py
# Purpose: Main script to train a probe model, calculating embeddings online.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader # Only DataLoader needed here
import os
import sys
import yaml
import argparse
from datetime import datetime
import numpy as np # For checking embedding dim later

# Import helpers from other project files
try:
    # Import the data loader function and the embedding function
    from data_preprocess import get_image_dataloader, extract_embedding, split_dataset_into_dataloaders
    # Import the probe network definition
    from nn import LinearProbe
    # Import the modified training loop helpers
    from train import train_one_epoch, evaluate
except ImportError as e:
    print(f"Error importing helper modules (data_preprocess.py, nn.py, train.py): {e}")
    print("Ensure these files are in the same directory or Python path.")
    sys.exit(1)

# Import Hugging Face classes needed here for model loading/dim check
try:
     from transformers import AutoProcessor, AutoModel
except ImportError:
     print("Error: `transformers` library not installed. Please install it.")
     sys.exit(1)


# --- Argument Parsing for Config File ---
parser = argparse.ArgumentParser(description="Train a probe model on vision model features (online embedding extraction).")
parser.add_argument('--config_file', type=str, default='config.yaml', # Point to new config file
                    help='Path to the YAML configuration file for probing')
args = parser.parse_args()

# --- Load Configuration ---
config = None
probe_config = None
try:
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    config_path = os.path.join(script_dir, args.config_file)
    if not os.path.exists(config_path): raise FileNotFoundError(f"Config not found: {config_path}")

    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f: config = yaml.safe_load(f) or {}

    probe_config = config.get('ProbeTraining')
    if not probe_config or not isinstance(probe_config, dict):
        raise ValueError("Section 'ProbeTraining' not found or invalid in config.")

    # --- Extract Parameters ---
    PROBING_DATASET_NAME = probe_config.get('PROBING_DATASET_NAME')
    DATASET_PATH = probe_config.get('DATASET_PATH') # Path to ImageFolder root
    VISION_MODEL_ID = probe_config.get('VISION_MODEL_ID')
    IMAGE_SIZE = int(probe_config.get('IMAGE_SIZE', 224))
    EPOCHS = int(probe_config.get('EPOCHS', 10))
    BATCH_SIZE = int(probe_config.get('BATCH_SIZE', 64)) # Keep batch size reasonable
    LEARNING_RATE = float(probe_config.get('LEARNING_RATE', 0.001))
    OPTIMIZER_NAME = probe_config.get('OPTIMIZER', 'Adam')
    OUTPUT_DIR_PROBE_BASE = probe_config.get('OUTPUT_DIR_PROBE_BASE', 'probe_results_online')
    NUM_WORKERS = int(probe_config.get('NUM_WORKERS', 0))

    # Validate essential params
    if not all([PROBING_DATASET_NAME, DATASET_PATH, VISION_MODEL_ID]):
         raise ValueError("Missing essential keys in ProbeTraining config (DATASET_NAME, DATASET_PATH, VISION_MODEL_ID)")
    # Other checks...

except FileNotFoundError as e: print(f"Error: {e}"); sys.exit(1)
except (ValueError, KeyError, TypeError) as e: print(f"Error processing config '{args.config_file}': {e}"); sys.exit(1)
except Exception as e: print(f"An unexpected error occurred during setup: {e}"); sys.exit(1)


# --- Main Execution ---
if __name__ == "__main__":
    print("\n--- Probe Model Training (Online Embedding Extraction) ---")
    print(f"--- Using Vision Model: {VISION_MODEL_ID} ---")
    print(f"--- Probing Dataset:  {PROBING_DATASET_NAME} ({DATASET_PATH}) ---")

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load DataLoaders (to get num_classes and iterate images) ---
    print("Loading dataset and creating DataLoaders...")
    # Use is_train=False transforms for both, as augmentations might complicate direct comparison
    # Or apply train transforms only for train_loader if desired
    dataloader, num_classes = get_image_dataloader(
        data_path=DATASET_PATH,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        is_train=False,
        num_workers=NUM_WORKERS
    )
    if dataloader is None:
        print("DataLoader creation failed. Exiting test.")
        sys.exit(1)
    
    train_loader, val_loader = split_dataset_into_dataloaders(
        full_dataset=dataloader.dataset,
        train_split_ratio=0.8,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    print(f"DataLoader created successfully. Number of classes: {num_classes}")
    print(f"Train DataLoader: {len(train_loader.dataset)} samples")
    print(f"Validation DataLoader: {len(val_loader.dataset)} samples")
    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        print("ERROR: One of the DataLoaders is empty. Check dataset split.")
        sys.exit(1)

    if train_loader is None or val_loader is None:
        print("Failed to create DataLoaders. Check dataset path and structure.")
        sys.exit(1)

    NUM_CLASSES = num_classes # Use train num_classes
    print(f"Determined Number of Classes: {NUM_CLASSES}")
    print("DataLoaders ready.")

    # --- Load Vision Model & Processor ---
    print(f"Loading vision model: {VISION_MODEL_ID}")
    try:
        processor = AutoProcessor.from_pretrained(VISION_MODEL_ID)
        # Load model and ensure it's frozen
        vision_model = AutoModel.from_pretrained(VISION_MODEL_ID).to(device)
        vision_model.eval() # Set to eval mode
        for param in vision_model.parameters(): # Freeze weights
            param.requires_grad = False
        print(f"Loaded and froze vision model: {type(vision_model).__name__}")
    except Exception as e:
        print(f"Error loading vision model: {e}"); sys.exit(1)


    # --- Determine Embedding Dimension ---
    print("Determining embedding dimension...")
    embedding_dim = None
    try:
        # Get a sample batch
        sample_images, _ = next(iter(val_loader)) # Use val loader to avoid augmentations if any
        # Extract embedding for the sample
        sample_embedding = extract_embedding(sample_images[:1], vision_model, processor, device) # Use only one image
        if sample_embedding is None: raise ValueError("Failed to extract sample embedding.")
        print(f"Sample embedding shape: {sample_embedding.shape}")
        embedding_dim = sample_embedding.shape[-1] # Get last dimension size
        # Clear sample data from memory/GPU
        del sample_images, sample_embedding
        if 'cuda' in device.type: torch.cuda.empty_cache()
    except StopIteration:
         print("ERROR: Validation loader yielded no batches to determine embedding dim.")
         sys.exit(1)
    except Exception as e:
        print(f"Error determining embedding dimension: {e}"); sys.exit(1)


    # --- Initialize Probe Model, Loss, Optimizer ---
    print(f"Initializing Linear probe...")
    probe_model = LinearProbe(embedding_dim=embedding_dim, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    if OPTIMIZER_NAME.lower() == 'adam':
        optimizer = optim.Adam(probe_model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER_NAME.lower() == 'sgd':
        optimizer = optim.SGD(probe_model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    else:
         print(f"ERROR: Unsupported OPTIMIZER '{OPTIMIZER_NAME}'."); sys.exit(1)

    print(f"Probe Model:\n{probe_model}")
    print(f"Criterion: {type(criterion).__name__}, Optimizer: {type(optimizer).__name__}")

    # --- Prepare Output Directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_id_name = VISION_MODEL_ID.replace('/', '_').replace('\\', '_')
    run_output_dir_specific = os.path.join(OUTPUT_DIR_PROBE_BASE,
                                          f"{PROBING_DATASET_NAME}_{safe_model_id_name}",
                                          f"probe_linear_lr{LEARNING_RATE}_{timestamp}")
    os.makedirs(run_output_dir_specific, exist_ok=True)
    print(f"Probe results will be saved to: {run_output_dir_specific}")

    # --- Training Loop ---
    print(f"\n--- Starting Training (VERY SLOW - Online Embedding Extraction) ---")
    print(f"--- Running for {EPOCHS} epochs ---")
    best_val_acc = 0.0
    # History tracking can be added here if needed

    for epoch in range(EPOCHS):
        # Train - pass vision model and processor to helper
        train_loss, train_acc = train_one_epoch(
            model = probe_model,
            dataloader = train_loader,
            criterion = criterion,
            optimizer = optimizer,
            device = device,
            extract_embedding = extract_embedding,
            vision_model = vision_model,
            processor = processor,
            epoch_num = epoch,
            total_epochs = EPOCHS
        )

        # Evaluate - pass vision model and processor to helper
        val_loss, val_acc = evaluate(
            model = probe_model,
            dataloader = val_loader,
            criterion = criterion,
            device = device
        )
        print(f"Epoch {epoch+1}/{EPOCHS} [Train]  Loss: {train_loss:.4f} Acc: {train_acc:.2f}%")
        print(f"Epoch {epoch+1}/{EPOCHS} [Val]   Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            print(f"  New best validation accuracy: {val_acc:.2f}%")
            best_val_acc = val_acc
            checkpoint_path = os.path.join(run_output_dir_specific, 'best_model.pth')
            torch.save({ "/* ... save state ... */" }, checkpoint_path) # Add details to save
            print(f"  Saved best model checkpoint.")
        print("-" * 30)

    print("--- Training Finished ---")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    # Add saving final model, plots etc. here using run_output_dir_specific
    final_model_path = os.path.join(run_output_dir_specific, 'final_model.pth')
    torch.save(probe_model.state_dict(), final_model_path)
    print(f"Saved final model state_dict to {final_model_path}")
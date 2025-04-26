# Filename: main.py
# Purpose: Main script to train a probe model on PRE-COMPUTED embeddings.
# Loads embeddings saved by extract_embeddings.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
import yaml
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Import helpers
try:
    from nn import LinearProbe
    from train import train_one_epoch, evaluate # Use the reverted train.py helpers
except ImportError as e:
    print(f"Error importing helper modules (nn.py, train.py): {e}")
    sys.exit(1)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Train probe on pre-computed embeddings.")
parser.add_argument('--config_file', type=str, default='config.yaml',
                    help='Path to the YAML configuration file for probing')
# Allow overriding key params if needed, useful for batch probing later
parser.add_argument('--vision_model_id', type=str, default=None, help='Override VISION_MODEL_ID from config (must match embedding path)')
parser.add_argument('--dataset_name', type=str, default=None, help='Override PROBING_DATASET_NAME from config (must match embedding path)')
parser.add_argument('--embedding_path', type=str, default=None, help='Override EMBEDDING_BASE_PATH from config')
parser.add_argument('--output_dir', type=str, default=None, help='Override OUTPUT_DIR_PROBE_BASE from config')

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
    probe_config = config.get('ProbeTraining', {})

    # --- Determine Effective Parameters ---
    VISION_MODEL_ID = args.vision_model_id or probe_config.get('VISION_MODEL_ID')
    PROBING_DATASET_NAME = args.dataset_name or probe_config.get('PROBING_DATASET_NAME')
    EMBEDDING_BASE_PATH = args.embedding_path or probe_config.get('EMBEDDING_BASE_PATH')
    OUTPUT_DIR_PROBE_BASE = args.output_dir or probe_config.get('OUTPUT_DIR_PROBE_BASE', 'probe_results')
    # Hyperparameters from config
    EPOCHS = int(probe_config.get('EPOCHS', 10))
    BATCH_SIZE = int(probe_config.get('BATCH_SIZE', 128))
    LEARNING_RATE = float(probe_config.get('LEARNING_RATE', 0.001))
    OPTIMIZER_NAME = probe_config.get('OPTIMIZER', 'Adam')
    NUM_WORKERS = int(probe_config.get('PROBE_NUM_WORKERS', 0)) # For loading embeddings

    # Validate essential params
    if not all([PROBING_DATASET_NAME, VISION_MODEL_ID, EMBEDDING_BASE_PATH, OUTPUT_DIR_PROBE_BASE]):
        raise ValueError("Missing essential config/args: DATASET_NAME, VISION_MODEL_ID, EMBEDDING_BASE_PATH, OUTPUT_DIR_PROBE_BASE")

except FileNotFoundError as e: print(f"Error: {e}"); sys.exit(1)
except (ValueError, KeyError, TypeError) as e: print(f"Error processing config '{args.config_file}': {e}"); sys.exit(1)
except Exception as e: print(f"An unexpected error occurred during setup: {e}"); sys.exit(1)


# --- Main Execution ---
if __name__ == "__main__":
    print("\n--- Probe Model Training (Using Pre-Computed Embeddings) ---")
    print(f"--- Probing Features From: {VISION_MODEL_ID} ---")
    print(f"--- On Dataset:            {PROBING_DATASET_NAME} ---")

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Construct Paths and Load Pre-Computed Embeddings ---
    print("Constructing embedding paths...")
    safe_model_id_name = VISION_MODEL_ID.replace('/', '_').replace('\\', '_')
    embedding_dir = os.path.join(EMBEDDING_BASE_PATH, safe_model_id_name)
    train_embed_file = os.path.join(embedding_dir, f"{PROBING_DATASET_NAME}_train_embeddings.pt")
    train_label_file = os.path.join(embedding_dir, f"{PROBING_DATASET_NAME}_train_labels.pt")
    val_embed_file = os.path.join(embedding_dir, f"{PROBING_DATASET_NAME}_val_embeddings.pt")
    val_label_file = os.path.join(embedding_dir, f"{PROBING_DATASET_NAME}_val_labels.pt")

    print(f"Attempting to load embeddings from: {embedding_dir}")
    try:
        train_embeddings = torch.load(train_embed_file, map_location='cpu')
        train_labels = torch.load(train_label_file, map_location='cpu')
        val_embeddings = torch.load(val_embed_file, map_location='cpu')
        val_labels = torch.load(val_label_file, map_location='cpu')
        print("Embeddings and labels loaded successfully.")

        # --- Determine Dimensions Automatically ---
        embedding_dim = train_embeddings.shape[1]
        # Infer num_classes (safer: check both train/val or get from dataset earlier)
        num_classes = len(torch.unique(train_labels))
        print(f"Determined Embedding Dimension: {embedding_dim}")
        print(f"Determined Number of Classes: {num_classes}")

    except FileNotFoundError as e:
        print(f"\nERROR: Pre-computed embedding file not found: {e}")
        print("Please ensure you have run the 'extract_embeddings.py' script first")
        print(f"and that files exist in the expected directory: '{embedding_dir}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading embeddings/labels: {e}"); sys.exit(1)

    # --- Create DataLoaders for Embeddings ---
    print("Creating DataLoaders for probe training...")
    train_dataset = TensorDataset(train_embeddings, train_labels)
    val_dataset = TensorDataset(val_embeddings, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))
    print("DataLoaders ready.")

    # --- Initialize Model, Loss, Optimizer ---
    print(f"Initializing Linear probe...")
    probe_model = LinearProbe(embedding_dim=embedding_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    if OPTIMIZER_NAME.lower() == 'adam':
        optimizer = optim.Adam(probe_model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER_NAME.lower() == 'sgd':
        optimizer = optim.SGD(probe_model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    else: print(f"ERROR: Unsupported OPTIMIZER '{OPTIMIZER_NAME}'."); sys.exit(1)
    print(f"Model:\n{probe_model}")
    print(f"Criterion: {type(criterion).__name__}, Optimizer: {type(optimizer).__name__}")

    # --- Prepare Output Directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir_specific = os.path.join(OUTPUT_DIR_PROBE_BASE,
                                        f"{PROBING_DATASET_NAME}_{safe_model_id_name}",
                                        f"probe_linear_lr{LEARNING_RATE}_epoch{EPOCHS}_{timestamp}")
    os.makedirs(run_output_dir_specific, exist_ok=True)
    print(f"Probe results will be saved to: {run_output_dir_specific}")

    # --- Training Loop (using helpers from train.py v1) ---
    print(f"\nStarting training for {EPOCHS} epochs...")
    best_val_acc = 0.0
    # History tracking lists can be added here...
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(EPOCHS):
        # Train (passes embeddings directly)
        train_loss, train_acc = train_one_epoch(
            probe_model, train_loader, criterion, optimizer, device, epoch, EPOCHS
        )
        # Evaluate (passes embeddings directly)
        val_loss, val_acc = evaluate(
            probe_model, val_loader, criterion, device
        )
        print(f"Epoch {epoch+1}/{EPOCHS} [Val]   Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            print(f"  New best validation accuracy: {val_acc:.2f}%")
            best_val_acc = val_acc
            checkpoint_path = os.path.join(run_output_dir_specific, 'best_model.pth')
            torch.save({
                'model_state_dict': probe_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_acc': best_val_acc
            }, checkpoint_path)
            print(f"  Saved best model checkpoint.")
        print("-" * 30)

    print("--- Training Finished ---")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    # Add saving final model, plots etc.
    final_model_path = os.path.join(run_output_dir_specific, 'final_model.pth')
    torch.save(probe_model.state_dict(), final_model_path)
    print(f"Saved final model state_dict to {final_model_path}")
    # Add plotting code here if desired, similar to previous version...
    # save the loss plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_output_dir_specific, 'training_history.png'))
    print(f"Saved training history plot to {run_output_dir_specific}/training_history.png")
    print("Training history plot saved.")
    # save best accuracy and loss to a text file
    with open(os.path.join(run_output_dir_specific, 'training_summary.txt'), 'w') as f:
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Final Training Accuracy: {history['train_acc'][-1]:.2f}%\n")
        f.write(f"Final Training Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Validation Loss: {history['val_loss'][-1]:.4f}\n")
    print(f"Saved training summary to {run_output_dir_specific}/training_summary.txt")
    # Optionally, save the entire history to a file
    history_file = os.path.join(run_output_dir_specific, 'training_history.npy')
    np.save(history_file, history)
    print(f"Saved training history to {history_file}")
    print("All done! Check the output directory for results.")
    print("Exiting...")
    sys.exit(0)
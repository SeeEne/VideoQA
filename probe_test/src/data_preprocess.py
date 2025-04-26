# Filename: data_preprocess.py
# Purpose: Helper functions for data loading, image preprocessing (geometric),
#          and a flexible embedding extraction utility.

import os
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset, random_split
from torch import Generator
from transformers import AutoImageProcessor, AutoModel


# Note: Normalization constants are removed, as normalization should be
# handled by the specific model's Hugging Face processor.

def get_image_transforms(image_size=224, is_train=False):
    """
    Returns image transformations: Resize, Crop, ToTensor (scales to [0, 1]).
    Normalization should be handled by the model-specific processor later.
    Includes basic augmentation for training set if specified.
    """
    transform_list = []
    if is_train:
        # Basic training augmentations
        transform_list.extend([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        # Standard validation/test transforms (resize shortest side, center crop)
        # Resize slightly larger first to prevent aspect ratio distortion before crop
        transform_list.extend([
            transforms.Resize(int(image_size * 256 / 224)), # Standard practice: resize to 256 for 224 crop
            transforms.CenterCrop(image_size),
        ])

    transform_list.append(transforms.ToTensor()) # Converts PIL Image (HWC) [0, 255] to Tensor (CHW) [0, 1]

    return transforms.Compose(transform_list)

def get_image_dataloader(data_path, batch_size=32, image_size=224, is_train=False, num_workers=4, **kwargs):
    """
    Creates a DataLoader for image classification datasets (assumes ImageFolder structure).
    Applies geometric transforms and ToTensor, but *not* normalization.

    Args:
        data_path (str): Path to the root directory of the dataset (e.g., 'path/to/imagenet/train').
                         Assumes subdirectories named by class.
        batch_size (int): Number of samples per batch.
        image_size (int): The target size for image input (e.g., 224 for ViT-Base).
        is_train (bool): If True, applies training augmentations and shuffles.
        num_workers (int): Number of subprocesses to use for data loading.
        **kwargs: Additional arguments passed to the DataLoader constructor.

    Returns:
        torch.utils.data.DataLoader | None: The configured DataLoader, or None if error.
        int | None: Number of classes found, or None if error.
    """
    if not os.path.isdir(data_path):
        print(f"ERROR: Data directory not found: {data_path}")
        return None, None

    try:
        transform = get_image_transforms(image_size=image_size, is_train=is_train)
        dataset = datasets.ImageFolder(data_path, transform=transform)
        num_classes = len(dataset.classes)

        print(f"Found {len(dataset)} images in {num_classes} classes at '{data_path}'. Using Image Size: {image_size}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train, # Shuffle only for training
            num_workers=num_workers,
            pin_memory=True, # Helps speed up CPU-to-GPU transfer if using CUDA
            drop_last=is_train, # Drop last incomplete batch only during training
            **kwargs
        )
        return dataloader, num_classes
    except FileNotFoundError: # Might happen if ImageFolder finds no images
        print(f"ERROR: No images found in subdirectories of {data_path}")
        return None, None
    except Exception as e:
        print(f"ERROR creating DataLoader for {data_path}: {e}")
        return None, None

def split_dataset_into_dataloaders(
    full_dataset: Dataset,
    train_split_ratio: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 4,
    generator_seed: int = 42, # For reproducible splits
    **kwargs):
    """
    Splits a PyTorch Dataset into training and validation/test DataLoaders.

    Args:
        full_dataset (Dataset): The complete dataset instance to split
                                (e.g., ImageFolder, TensorDataset).
        train_split_ratio (float): Proportion of data to use for training (e.g., 0.8 for 80%).
        batch_size (int): Batch size for the new DataLoaders.
        num_workers (int): Number of workers for the new DataLoaders.
        generator_seed (int): Seed for the random split for reproducibility.
        **kwargs: Additional arguments passed to the DataLoader constructors (e.g., pin_memory).

    Returns:
        tuple[DataLoader | None, DataLoader | None]: A tuple containing the
                                                    (train_dataloader, val_dataloader).
                                                    Returns (None, None) if splitting fails.
    """
    if not isinstance(full_dataset, Dataset):
        print("Error: Input must be a PyTorch Dataset instance.")
        return None, None
    if not 0 < train_split_ratio < 1:
        print("Error: train_split_ratio must be between 0 and 1.")
        return None, None

    try:
        dataset_size = len(full_dataset)
        train_size = int(dataset_size * train_split_ratio)
        val_size = dataset_size - train_size

        if train_size == 0 or val_size == 0:
             print(f"Error: Split results in empty dataset (Train: {train_size}, Val: {val_size}). Adjust ratio or check dataset size.")
             return None, None

        print(f"Splitting dataset (size {dataset_size}) into Train ({train_size}) / Val ({val_size})")

        # Use a generator for reproducibility
        generator = Generator().manual_seed(generator_seed)
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

        # Create DataLoaders for each subset
        # NOTE: If you applied specific training transforms to full_dataset,
        # you might ideally want different transforms for train/val subsets here.
        # This implementation assumes the transform applied to full_dataset is suitable,
        # or that full_dataset contains data already transformed (e.g., embeddings).
        # If full_dataset has *basic* transforms (Resize/Crop/ToTensor), this is usually fine.
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True, # Shuffle training data
            num_workers=num_workers,
            pin_memory=kwargs.get('pin_memory', True), # Default pin_memory=True
            drop_last=True, # Often drop last incomplete batch for training
            **{k: v for k, v in kwargs.items() if k != 'pin_memory'} # Pass other kwargs
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False, # No need to shuffle validation/test data
            num_workers=num_workers,
            pin_memory=kwargs.get('pin_memory', True),
            drop_last=False,
             **{k: v for k, v in kwargs.items() if k != 'pin_memory'}
        )

        print("Train and validation DataLoaders created.")
        return train_loader, val_loader

    except Exception as e:
        print(f"Error during dataset split or DataLoader creation: {e}")
        return None, None

# --- Embedding Extraction Helper ---
def extract_embedding(image_batch_tensor, model, processor, device):
    """
    Extracts image embeddings from a batch of *pre-transformed* image tensors.
    Uses the provided processor to apply model-specific normalization and prepares model inputs.
    Implements fallback logic for different model APIs (CLIP, SigLIP, BLIP, DINOv2 etc.).

    Args:
        image_batch_tensor (torch.Tensor): Batch of image tensors (output of DataLoader using
                                           get_image_transforms), expected range [0, 1].
        model (torch.nn.Module): The loaded Hugging Face vision model (e.g., CLIPModel, SiglipModel,
                                 BlipModel, AutoModel). Assumed to be on the correct device.
        processor (object): The corresponding Hugging Face processor (e.g., CLIPProcessor,
                            AutoImageProcessor). Handles normalization.
        device (torch.device): The device the model is on ('cuda' or 'cpu').

    Returns:
        torch.Tensor | None: A tensor containing the extracted embeddings [batch_size, embedding_dim]
                              moved to CPU, or None if extraction fails.
    """
    # Ensure model is in eval mode and inputs are on the correct device
    model.eval()
    # Input tensor should already be on the device if DataLoader uses pin_memory with CUDA,
    # but explicitly moving is safer.
    inputs_on_device = image_batch_tensor.to(device)

    image_features = None # Initialize

    try:
        with torch.no_grad():
            # 1. Prepare model inputs using the processor
            # The processor handles the model-specific normalization using its internal stats.
            # It expects input tensors in [0, 1] range typically.
            # Use `images` argument for image-only processors/models.
            # Use `pixel_values` if the model expects that key specifically. Check processor docs.
            # Let's try a flexible approach using common patterns.
            try:
                # Most processors work with 'images' argument
                inputs = processor(images=inputs_on_device, return_tensors="pt", do_rescale=False).to(device)
            except TypeError as e:
                # Some might strictly expect 'pixel_values' as the tensor itself
                # This path is less likely if input is already a tensor
                print(f"Info: Processor might expect direct tensor via pixel_values? Trying that. Error was: {e}")
                inputs = {"pixel_values": inputs_on_device} # Assume model takes dict

            # 2. Extract Features (with fallback logic)
            # Strategy 1: Try get_image_features (Common in CLIP, SigLIP, some BLIP)
            if hasattr(model, 'get_image_features'):
                image_features = model.get_image_features(**inputs)
                # Handle potential sequence output (e.g., from BLIP) -> Use CLS token
                if isinstance(image_features, torch.Tensor) and image_features.ndim == 3:
                    print(f"Info: get_image_features returned sequence (shape {image_features.shape}), using CLS token.")
                    image_features = image_features[:, 0] # Assumes CLS token is at index 0

            # Strategy 2: Try standard forward pass and CLS token (Common in ViT/DINOv2)
            if image_features is None and hasattr(model, 'forward'):
                outputs = model(**inputs) # Use standard forward call
                if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                    if outputs.last_hidden_state.ndim == 3 and outputs.last_hidden_state.shape[1] > 0:
                        print(f"Info: Using CLS token from last_hidden_state.")
                        image_features = outputs.last_hidden_state[:, 0] # CLS token
                    else:
                        print(f"Warning: last_hidden_state has unexpected shape: {outputs.last_hidden_state.shape}")


            # Strategy 3: Try standard forward pass and pooler_output
            if image_features is None and hasattr(model, 'forward'):
                # Check if 'outputs' exists from previous step
                if 'outputs' not in locals(): outputs = model(**inputs)
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    print(f"Info: Using pooler_output.")
                    image_features = outputs.pooler_output

            # --- Check if we got features ---
            if image_features is None:
                raise NotImplementedError(f"Could not extract features using known methods for model {type(model).__name__}.")

            # --- Final L2 Normalization ---
            if isinstance(image_features, torch.Tensor):
                # Ensure features are float32 before normalization for stability
                image_features = image_features.float()
                image_features = image_features / (image_features.norm(p=2, dim=-1, keepdim=True) + 1e-6)
            else:
                raise TypeError(f"Extracted features are not a Tensor, but {type(image_features)}.")

        return image_features.cpu() # Return features on CPU

    except Exception as e:
        model_name = type(model).__name__ if model else "None"
        # Provide more context in the error message
        print(f"ERROR during embedding extraction with {model_name} for an image batch.")
        print(f"  Input tensor shape: {image_batch_tensor.shape}, Device: {image_batch_tensor.device}")
        print(f"  Error details: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None

# --- Example Usage (Illustrative - would be in other scripts) ---
if __name__ == '__main__':
    print("\n--- Testing data_preprocess.py ---")

    # --- Configuration for Test ---
    # <<< IMPORTANT: Set this path to where you downloaded and structured EuroSAT >>>
    test_data_path = "../dataset/EuroSAT" # Assumes EuroSAT class folders are inside this directory
    test_model_id = "openai/clip-vit-base-patch32" # Example model to test with
    test_batch_size = 4
    test_image_size = 224 # Standard for many ViTs
    test_num_workers = 0 # Use 0 for main process to simplify debugging, set higher (e.g., 2 or 4) for speed

    # Check if test data directory exists
    if not os.path.isdir(test_data_path):
        print("-" * 50)
        print(f"ERROR: Test data directory not found at '{test_data_path}'")
        print("Please download the EuroSAT dataset, unzip it, and place the class folders")
        print("(AnnualCrop, Forest, etc.) directly inside the 'eurosat_data' directory")
        print("in the same location as this script, or update test_data_path.")
        print("-" * 50)
        sys.exit(1)

    # --- 1. Test DataLoader ---
    print(f"\nTesting DataLoader with path: {test_data_path}")
    # Use is_train=False for test/validation transforms
    dataloader, num_classes = get_image_dataloader(
        data_path=test_data_path,
        batch_size=test_batch_size,
        image_size=test_image_size,
        is_train=False,
        num_workers=test_num_workers
    )

    if dataloader is None:
        print("DataLoader creation failed. Exiting test.")
        sys.exit(1)

    print(f"DataLoader created successfully. Number of classes: {num_classes}")

    # --- 2. Test Embedding Extraction (requires transformers and torch) ---
    print(f"\nTesting Embedding Extraction with model: {test_model_id}")
    try:
        

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load model and processor
        print("Loading model and processor...")
        processor = AutoImageProcessor.from_pretrained(test_model_id)
        model = AutoModel.from_pretrained(test_model_id).to(device)
        model.eval() # Set to evaluation mode
        print("Model and processor loaded.")

        # Get one batch from the dataloader
        print("Fetching one batch...")
        try:
            image_batch, label_batch = next(iter(dataloader))
            print(f"Image batch shape: {image_batch.shape}, Type: {image_batch.dtype}") # Should be [B, 3, H, W], float32 [0,1]
            print(f"Label batch shape: {label_batch.shape}, Type: {label_batch.dtype}")
        except StopIteration:
            print("ERROR: DataLoader yielded no batches. Is the dataset empty or path correct?")
            sys.exit(1)
        except Exception as e:
             print(f"ERROR fetching batch from DataLoader: {e}")
             sys.exit(1)


        # Extract embeddings for the batch
        print("Extracting embeddings for the batch...")
        embeddings = extract_embedding(image_batch, model, processor, device)

        if embeddings is not None:
            print(f"Successfully extracted embeddings!")
            print(f"  Output embedding shape: {embeddings.shape}") # Should be [batch_size, embedding_dim]
            print(f"  Output embedding type: {embeddings.dtype}") # Should be float32 (CPU tensor)
            # Optionally print first few values
            # print("  Example embedding (first vector):\n", embeddings[0][:10])
        else:
            print("Embedding extraction failed for the test batch.")

    except ImportError:
        print("\nWARNING: `transformers` library not installed.")
        print("Please install it (`pip install transformers[torch]`) to run the embedding extraction test.")
    except Exception as e:
        print(f"\nAn error occurred during the embedding extraction test: {e}")
        import traceback
        traceback.print_exc()


    # Test for the training and validation split
    print("\nTesting dataset split into DataLoaders...")
    train_loader, val_loader = split_dataset_into_dataloaders(
        full_dataset=dataloader.dataset,
        train_split_ratio=0.8,
        batch_size=test_batch_size,
        num_workers=test_num_workers
    )
    
    # convert into embedding
    print(f"Train DataLoader: {len(train_loader.dataset)} samples")
    print(f"Validation DataLoader: {len(val_loader.dataset)} samples")
    embeddings_train = extract_embedding(train_loader.dataset[0][0], model, processor, device)
    if embeddings_train is not None:
        print(f"Successfully extracted embeddings for training set!")
        print(f"  Output embedding shape: {embeddings_train.shape}") # Should be [batch_size, embedding_dim]
        print(f"  Output embedding type: {embeddings_train.dtype}") # Should be float32 (CPU tensor)
    else:
        print("Embedding extraction failed for the training set.")
    embeddings_val = extract_embedding(val_loader.dataset[0][0], model, processor, device)
    if embeddings_val is not None:
        print(f"Successfully extracted embeddings for validation set!")
        print(f"  Output embedding shape: {embeddings_val.shape}") # Should be [batch_size, embedding_dim]
        print(f"  Output embedding type: {embeddings_val.dtype}") # Should be float32 (CPU tensor)
    else:
        print("Embedding extraction failed for the validation set.")
    print("\n--- Test Complete ---")
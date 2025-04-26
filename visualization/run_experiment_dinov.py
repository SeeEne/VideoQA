# Filename: run_experiment_v5.py
# Purpose: Runs Minimal Change Sensitivity Analysis using DINOv2 models.
# Loads images generated across different sizes, performs analysis,
# and plots results against shape size.
# ------------------------------------------------------

import torch
import numpy as np
from PIL import Image
import os
import sys
from tqdm.notebook import tqdm # Or from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine as cosine_distance
import lpips # Still needed for visual distance

# --- Import specific Hugging Face classes ---
from transformers import AutoImageProcessor, AutoModel

import collections
import re
import yaml
import argparse

# --- Argument Parsing ---
# Note: Renamed --clip_model to --vision_model_id for clarity
parser = argparse.ArgumentParser(description="Run DINOv2 Minimal Change Sensitivity Analysis")
parser.add_argument('--vision_model_id', type=str, default=None, # Default is None
                    help='Override VISION_MODEL_ID from config.yaml (e.g., facebook/dinov2-base)')
parser.add_argument('--lpips_net', type=str, default=None, # Default is None
                    help='Override LPIPS_NET_TYPE from config.yaml')
parser.add_argument('--output_dir', type=str, default=None, # Default is None
                    help='Override OUTPUT_DIR from config.yaml (specifies where results for this specific run go)')
parser.add_argument('--config_file', type=str, default='config.yaml',
                    help='Path to the YAML configuration file')
args = parser.parse_args()

# --- Configuration Loading ---
config = None
config_path = None
try:
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    config_path = os.path.join(script_dir, args.config_file)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        if config is None: config = {}
except FileNotFoundError as e:
    print(f"Error: {e}")
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing YAML file '{args.config_file}': {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred loading config: {e}")
    sys.exit(1)

# --- Determine Effective Configuration ---
try:
    exp_config = config.get('Experiment', {})
    # Use VISION_MODEL_ID key now
    cfg_vision_model = exp_config.get('VISION_MODEL_ID')
    cfg_lpips_net = exp_config.get('LPIPS_NET_TYPE')
    cfg_output_dir = exp_config.get('OUTPUT_DIR', 'results_default_dino')
    cfg_base_stimuli_dir = exp_config.get('BASE_STIMULI_DIR')
    cfg_shape_params = exp_config.get('ShapeSizeParams')

    # Override with args if provided
    VISION_MODEL_ID = args.vision_model_id if args.vision_model_id is not None else cfg_vision_model
    LPIPS_NET_TYPE = args.lpips_net if args.lpips_net is not None else cfg_lpips_net
    RUN_OUTPUT_DIR = args.output_dir if args.output_dir is not None else cfg_output_dir
    BASE_STIMULI_DIR = cfg_base_stimuli_dir
    shape_params = cfg_shape_params

    # Validation
    missing = []
    if not VISION_MODEL_ID: missing.append("Vision Model ID (config/arg)")
    if not LPIPS_NET_TYPE: missing.append("LPIPS Net Type (config/arg)")
    if not RUN_OUTPUT_DIR: missing.append("Output Directory (config/arg)")
    if not BASE_STIMULI_DIR: missing.append("BASE_STIMULI_DIR (config)")
    if not shape_params: missing.append("ShapeSizeParams (config)")
    if missing:
        print(f"ERROR: Missing required parameters: {', '.join(missing)}")
        sys.exit(1)

    # Generate SHAPE_SIZES_TESTED
    SHAPE_SIZES_TESTED = None
    try:
        low = shape_params.get('lowest_size')
        high = shape_params.get('highest_size')
        interval = shape_params.get('interval')
        decimals = shape_params.get('rounding_decimals')
        if not all(isinstance(p, (int, float)) for p in [low, high, interval, decimals]):
            raise ValueError("Missing or non-numeric shape size parameters")
        SHAPE_SIZES_TESTED = np.round(np.arange(low, high, interval), int(decimals))
        if SHAPE_SIZES_TESTED.size == 0:
            raise ValueError(f"Shape size parameters resulted in empty range")
    except Exception as e:
        print(f"Error processing ShapeSizeParams: {e}")
        sys.exit(1)

except KeyError as e:
    print(f"Error: Missing key in config structure: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred processing config: {e}")
    sys.exit(1)

# --- Technical Setup ---
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
     device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# --- Feature Extraction Function for DINOv2 --- <<< MODIFIED HERE >>>
def get_image_embedding(image_path, model, processor, device):
    """
    Loads image and extracts DINOv2 image embedding (CLS token).
    """
    try:
        image = Image.open(image_path).convert("RGB")
        # Use AutoImageProcessor - typically expects pixel_values
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # DINOv2 standard practice: Use the CLS token embedding
            # It's the first token in the last_hidden_state sequence
            # Shape: [batch_size, sequence_length, hidden_dimension]
            image_features = outputs.last_hidden_state[:, 0] # Select CLS token

        # Apply L2 normalization (optional but good for cosine distance)
        image_features = image_features / (image_features.norm(p=2, dim=-1, keepdim=True) + 1e-6)
        return image_features.cpu().numpy().squeeze()

    except Exception as e:
        model_name = type(model).__name__ if model else "None"
        print(f"Error processing {image_path} with {model_name}: {e}")
        return None

# --- LPIPS Helper Function (Unchanged) ---
# ... (get_lpips_distance definition) ...
def get_lpips_distance(img_path1, img_path2, lpips_model, device):
    """Calculates LPIPS distance between two image files."""
    try:
        img1_tensor = lpips.im2tensor(lpips.load_image(img_path1)).to(device)
        img2_tensor = lpips.im2tensor(lpips.load_image(img_path2)).to(device)
        with torch.no_grad():
            distance = lpips_model(img1_tensor, img2_tensor).item()
        return distance
    except Exception as e:
        print(f"Error calculating LPIPS for {os.path.basename(img_path1)} vs {os.path.basename(img_path2)}: {e}")
        return None


# --- Main Experiment Script ---
if __name__ == "__main__":
    print("\n--- Module 2: Running DINOv2 Minimal Change Experiment ---") # Updated title
    print(f"--- Configuration Used ---")
    print(f"  Vision Model: {VISION_MODEL_ID}") # Changed label
    print(f"  LPIPS Net:    {LPIPS_NET_TYPE}")
    print(f"  Stimuli Base: {BASE_STIMULI_DIR}")
    print(f"  Output Dir:   {RUN_OUTPUT_DIR}")
    print(f"  Sizes Tested: {SHAPE_SIZES_TESTED}")
    print(f"--------------------------")

    run_output_path_full = os.path.abspath(RUN_OUTPUT_DIR)
    os.makedirs(run_output_path_full, exist_ok=True)
    print(f"Results will be saved to: {run_output_path_full}")

    # --- 1. Load Models --- <<< MODIFIED HERE >>>
    print("Loading models...")
    try:
        # Load DINOv2 model using AutoModel
        model = AutoModel.from_pretrained(VISION_MODEL_ID).to(device)
        # Load corresponding processor using AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(VISION_MODEL_ID)
        print(f"Loaded DINOv2 Model/Processor: {VISION_MODEL_ID}")
    except Exception as e:
        print(f"Error loading vision model '{VISION_MODEL_ID}': {e}. Exiting run.")
        sys.exit(1)

    try:
        # LPIPS loading remains the same
        lpips_model = lpips.LPIPS(net=LPIPS_NET_TYPE).to(device)
        lpips_model.eval()
        print(f"Loaded LPIPS Model: {LPIPS_NET_TYPE}")
    except Exception as e:
        print(f"Error loading LPIPS model '{LPIPS_NET_TYPE}': {e}. Exiting run.")
        sys.exit(1)
    print("Models loaded.")

    # --- 2. Process Results per Size ---
    results_by_size = collections.defaultdict(list)
    all_embeddings_cache = {} # Reuse cache logic
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    base_stimuli_path_full = os.path.join(script_dir, BASE_STIMULI_DIR)

    for current_shape_size in SHAPE_SIZES_TESTED:
        size_str = f"{current_shape_size:.2f}"
        current_stimuli_dir = os.path.join(base_stimuli_path_full, f"size_{size_str}")
        print(f"\n--- Processing Size: {size_str} ---")

        # Find files and pairs logic remains the same
        # ... (ensure this section is present and correct) ...
        all_files = [f for f in os.listdir(current_stimuli_dir) if f.endswith(".png")] # Example
        config_to_filepath = {} # Example
        pattern = re.compile(r"config_S(\d\.\d+)_+([a-z]+)_([a-z]+)_([a-z]+)\.png") # Example pattern
        for fname in all_files:
            match = pattern.match(fname)
            if match:
                # Basic check, assumes filename format is correct
                config_tuple = (match.group(2), match.group(3), match.group(4))
                config_to_filepath[config_tuple] = os.path.join(current_stimuli_dir, fname)
            else:
                print(f"Warning: Filename '{fname}' does not match expected pattern. Use defaulting to unknown config.")
                config_tuple = (fname, "unknown", "unknown")
                config_to_filepath[config_tuple] = os.path.join(current_stimuli_dir, fname)

        minimal_pairs = [] # Example
        configs = list(config_to_filepath.keys())
        for i in range(len(configs)):
            for j in range(i + 1, len(configs)):
                conf1 = configs[i]
                conf2 = configs[j]
                diff_count = sum(1 for k in range(len(conf1)) if conf1[k] != conf2[k])
                if diff_count == 1:
                    if conf1 in config_to_filepath and conf2 in config_to_filepath:
                        fp1 = config_to_filepath[conf1]
                        fp2 = config_to_filepath[conf2]
                        minimal_pairs.append((fp1, fp2, conf1, conf2))


        if not minimal_pairs:
            print(f"Warning: No minimal pairs found for size {size_str}. Skipping analysis.")
            continue

        current_size_paths = set(fp for pair in minimal_pairs for fp in pair[:2])

        print("Extracting embeddings...")
        for img_path in tqdm(list(current_size_paths), desc=f"Embeddings {size_str}", leave=False):
            if img_path not in all_embeddings_cache:
                # Call the updated function, passing DINOv2 model and processor
                embedding = get_image_embedding(img_path, model, processor, device)
                if embedding is not None:
                    all_embeddings_cache[img_path] = embedding

        print("Calculating distances...")
        size_results_list = []
        for fp1, fp2, conf1, conf2 in tqdm(minimal_pairs, desc=f"Distances {size_str}", leave=False):
            if fp1 in all_embeddings_cache and fp2 in all_embeddings_cache:
                E1 = all_embeddings_cache[fp1]
                E2 = all_embeddings_cache[fp2]
                d_image = cosine_distance(E1, E2)
                d_visual = get_lpips_distance(fp1, fp2, lpips_model, device)
                if d_visual is not None:
                    size_results_list.append({'d_image': d_image, 'd_visual': d_visual})
            else:
                print(f"Warning: Skipping pair for size {size_str} due to missing embeddings.")

        results_by_size[current_shape_size] = size_results_list
        print(f"Calculated distances for {len(size_results_list)} pairs for size {size_str}.")
        # --- End Size Loop ---

    # --- 3. Aggregate and Analyze Overall Results ---
    # ... (Aggregation logic is unchanged) ...
    avg_d_image_list = []
    avg_d_visual_list = []
    avg_ratio_list = []
    sizes_processed = sorted(results_by_size.keys())
    for size in sizes_processed:
        # ... (calculation of averages from size_results_list) ...
        size_results = results_by_size[size]
        if not size_results:
            # Append NaN or handle missing data appropriately
            avg_d_image_list.append(np.nan); avg_d_visual_list.append(np.nan); avg_ratio_list.append(np.nan)
            continue
        d_images = [r['d_image'] for r in size_results]
        d_visuals = [r['d_visual'] for r in size_results]
        ratios = [r['d_image'] / r['d_visual'] for r in size_results if r['d_visual'] > 1e-6]
        avg_d_image_list.append(np.mean(d_images))
        avg_d_visual_list.append(np.mean(d_visuals))
        avg_ratio_list.append(np.mean(ratios) if ratios else np.nan)


    # --- 4. Plotting ---
    # Plotting logic remains the same, ensure variable names match
    # Save filename should reflect the DINOv2 model ID
    print("Generating plots...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    # Plotting d_visual, d_image, R_swap using avg_*_list and sizes_processed...
    # (Ensure plotting code handles potential NaNs if some sizes failed)
    axs[0].plot(sizes_processed, avg_d_visual_list, marker='o', linestyle='-', color='blue', label='Avg LPIPS')
    axs[1].plot(sizes_processed, avg_d_image_list, marker='s', linestyle='-', color='red', label='Avg Cosine Dist (DINOv2)')
    axs[2].plot(sizes_processed, avg_ratio_list, marker='^', linestyle='-', color='green', label='Avg Ratio (R_swap)')
    axs[2].axhline(1.0, color='grey', linestyle='--', linewidth=0.8, label='Ratio = 1.0')

    axs[0].set_title('Average Visual Distance (LPIPS) vs. Shape Size')
    axs[0].set_ylabel('Avg LPIPS Distance (d_visual)')
    axs[0].grid(True); axs[0].legend()
    axs[1].set_title(f'Average Embedding Distance ({VISION_MODEL_ID}) vs. Shape Size')
    axs[1].set_ylabel('Avg Cosine Distance (d_image)')
    axs[1].grid(True); axs[1].legend()
    axs[2].set_title('Average Sensitivity Ratio (R_swap) vs. Shape Size')
    axs[2].set_ylabel('Avg Ratio (R_swap)')
    axs[2].set_xlabel('Shape Size (Blender Units)')
    axs[2].grid(True); axs[2].legend()
    plt.tight_layout()

    safe_vision_model_name = VISION_MODEL_ID.replace('/', '_').replace('\\', '_')
    safe_lpips_name = LPIPS_NET_TYPE.replace('/', '_').replace('\\', '_')
    plot_base_filename = f"size_sweep_analysis_{safe_vision_model_name}_{safe_lpips_name}.png"
    plot_filename = os.path.join(run_output_path_full, plot_base_filename)
    try:
        plt.savefig(plot_filename)
        print(f"Saved analysis plots to: {plot_filename}")
    except Exception as e:
        print(f"ERROR saving plot '{plot_filename}': {e}")
    plt.close(fig) # Close figure
    plt.show() # Keep show() for interactive runs maybe

    # --- 5. Final Interpretation --- (Unchanged)
    # ... (Print guidance for interpreting plots) ...

    print(f"\n--- Experiment Run Complete for {VISION_MODEL_ID} / {LPIPS_NET_TYPE} ---")
# Filename: run_batch.py
# Purpose: Runs the experiment analysis script (run_experiment_v3.py)
#          in parallel for different CLIP models and LPIPS types.

import yaml
import subprocess
import os
import itertools
import sys
from datetime import datetime
import concurrent.futures # For parallel execution
import tqdm # For progress bar

# --- Configuration ---
CONFIG_FILE = 'config.yaml'
# Assumes run_experiment_v3.py is modified to accept args
ANALYSIS_SCRIPT_NAME = 'run_experiment_v4.py'

# --- Worker Function ---
# This function will be executed by each worker process in the pool
def run_single_experiment(run_params):
    """
    Executes a single instance of the analysis script with given parameters.
    """
    clip_model, lpips_net, run_output_dir, config_file, script_name, run_id, total_runs = run_params

    print(f"\n--- [Worker {os.getpid()}] Starting Run {run_id}/{total_runs} ---")
    print(f"  CLIP Model: {clip_model}")
    print(f"  LPIPS Net:  {lpips_net}")
    print(f"  Results Dir: {run_output_dir}")

    command = [
        sys.executable, # Use the same python interpreter
        script_name,
        '--clip_model', clip_model,
        '--lpips_net', lpips_net,
        '--output_dir', run_output_dir,
        '--config_file', config_file
    ]

    print(f"  [Worker {os.getpid()}] Executing: {' '.join(command)}")
    try:
        # Run without capturing output - it will print directly to console
        result = subprocess.run(command, check=True, text=True, encoding='utf-8') # Keep text=True if needed by run() internals or for potential capture later
        print(f"  --- [Worker {os.getpid()}] Run {run_id} Completed Successfully (Return Code: {result.returncode}) ---")
        return (run_id, clip_model, lpips_net, True, None) # Success
    except subprocess.CalledProcessError as e:
        # Error messages from the script should have printed directly
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR: [Worker {os.getpid()}] Run {run_id} ({clip_model} / {lpips_net}) FAILED!")
        print(f"  Return Code: {e.returncode}")
        print(f"  Check console output above for error messages from the script.")
        # Optionally capture and return stderr if needed for debugging
        # stderr_output = e.stderr if hasattr(e, 'stderr') else 'stderr not captured'
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return (run_id, clip_model, lpips_net, False, f"Return Code: {e.returncode}") # Failure
    except Exception as e:
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR: [Worker {os.getpid()}] An unexpected error occurred launching run {run_id}: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return (run_id, clip_model, lpips_net, False, str(e)) # Failure

# --- Main Batch Execution ---
if __name__ == "__main__":
    print("--- Parallel Batch Experiment Runner ---")
    start_time = datetime.now()

    # --- 1. Load Config ---
    print(f"Loading configuration from: {CONFIG_FILE}")
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Config file '{CONFIG_FILE}' not found. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config file '{CONFIG_FILE}': {e}. Exiting.")
        sys.exit(1)

    # --- 2. Get Batch Parameters ---
    try:
        batch_config = config['BatchExperiment']
        exp_config = config['Experiment']

        results_base_dir = batch_config['RESULTS_BASE_DIR']
        clip_model_ids = batch_config['CLIP_MODEL_IDs']
        lpips_net_types = batch_config['LPIPS_NET_TYPEs']
        # <<< GET MAX WORKERS >>>
        max_workers = int(batch_config.get('MAX_WORKERS', 1)) # Default to 1 if not specified
        if max_workers <= 0:
            max_workers = 1
            print("Warning: MAX_WORKERS set to 0 or negative, defaulting to 1.")

        if not clip_model_ids or not lpips_net_types:
            print("ERROR: CLIP_MODEL_IDs or LPIPS_NET_TYPEs list is empty in config. Exiting.")
            sys.exit(1)

    except KeyError as e:
        print(f"Error: Missing key in '{CONFIG_FILE}' under BatchExperiment or Experiment: {e}. Exiting.")
        sys.exit(1)
    except (TypeError, ValueError) as e:
        print(f"Error parsing parameters in '{CONFIG_FILE}'. Check lists and MAX_WORKERS: {e}. Exiting.")
        sys.exit(1)


    # Ensure analysis script exists
    if not os.path.exists(ANALYSIS_SCRIPT_NAME):
        print(f"ERROR: Analysis script '{ANALYSIS_SCRIPT_NAME}' not found. Exiting.")
        sys.exit(1)

    # --- 3. Prepare Jobs ---
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    batch_results_base_path = os.path.join(script_dir, results_base_dir)
    os.makedirs(batch_results_base_path, exist_ok=True)
    print(f"Base directory for batch results: {batch_results_base_path}")
    print(f"Max parallel workers set to: {max_workers}")
    print(f"WARNING: Ensure you have enough VRAM for {max_workers} concurrent model loads!")

    jobs = []
    run_id_counter = 0
    combinations = list(itertools.product(clip_model_ids, lpips_net_types))
    total_runs = len(combinations)

    for clip_model, lpips_net in combinations:
        run_id_counter += 1
        safe_clip_name = clip_model.replace('/', '_').replace('\\', '_')
        safe_lpips_name = lpips_net.replace('/', '_').replace('\\', '_')
        run_output_dir = os.path.join(batch_results_base_path, f"{safe_clip_name}_{safe_lpips_name}")
        os.makedirs(run_output_dir, exist_ok=True) # Create dir before job starts

        # Package parameters for the worker function
        run_params = (clip_model, lpips_net, run_output_dir, CONFIG_FILE, ANALYSIS_SCRIPT_NAME, run_id_counter, total_runs)
        jobs.append(run_params)

    # --- 4. Execute Jobs in Parallel ---
    print(f"\nSubmitting {len(jobs)} jobs to Process Pool with {max_workers} workers...")
    successful_runs = 0
    failed_runs = 0
    futures = []

    # Use ProcessPoolExecutor to run jobs in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        for job_params in jobs:
            future = executor.submit(run_single_experiment, job_params)
            futures.append(future)

        # Process results as they complete
        print("\nWaiting for jobs to complete...")
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Batch Progress"):
            try:
                run_id, clip_model, lpips_net, success, error_msg = future.result()
                if success:
                    successful_runs += 1
                else:
                    failed_runs += 1
                    # Error details should have been printed by the worker's exception handler
            except Exception as exc:
                # Catch potential errors from the future.result() call itself
                failed_runs += 1
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"ERROR: A future completed with an unexpected exception: {exc}")
                # Attempt to find which job failed (might be hard if exception occurs before worker function starts properly)
                # This usually indicates a problem with pickling arguments or the executor itself.
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    # --- 5. Batch Summary ---
    end_time = datetime.now()
    print("\n--- Batch Execution Summary ---")
    print(f"Total Runs Attempted: {total_runs}")
    print(f"Successful Runs:      {successful_runs}")
    print(f"Failed Runs:          {failed_runs}")
    print(f"Max Workers:          {max_workers}")
    print(f"Total Execution Time: {end_time - start_time}")
    print(f"Results saved under:  {batch_results_base_path}")

    if failed_runs > 0:
        print("\nWARNING: Some runs failed. Please check the console output above for details.")

    print("--- Batch Complete ---")
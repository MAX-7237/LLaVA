import json
import os
import sys
import numpy as np

# Define lists for layers and samples to process
layer_indices = [1,2,3,4,5,7,8,9,10,11,13,14,15,16,17,19,20,21,22,23,25,26,27,28,29,31,32]
sample_indices = [0,1,2,3,4,5,6,7,8,9]

def process_single_case(layer_index, sample_index):
    # Input file path
    input_file = f"/data/users/Actor/LLaVA_Prune/results_lossing/results_lossing_{layer_index}/{sample_index}/loss_diff_sorted.json"
    
    # Output directories
    output_dir = os.path.dirname(input_file)
    pos_file = os.path.join(output_dir, "loss_diff_positive.json")
    neg_file = os.path.join(output_dir, "loss_diff_negative.json")

    print(f"\n{'='*50}")
    print(f"Processing: {input_file}")
    print(f"{'='*50}")

    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return

    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Data keys count: {len(data)}")

    # Initialize lists
    pos_list = [] # List of tuples: (index, abs_loss_diff, loss_diff)
    neg_list = []
    all_diffs = []

    # Iterate through the sorted JSON
    zero_count = 0
    total_tokens_processed = 0

    for loss_diff_str, idx in data.items():
        loss_diff = float(loss_diff_str)
        abs_diff = abs(loss_diff)
        all_diffs.append(loss_diff)
        total_tokens_processed += 1

        if loss_diff == 0:
            zero_count += 1

        # Handle if idx is a list (multiple tokens have same diff)
        if isinstance(idx, list):
            for single_idx in idx:
                if loss_diff > 0:
                    pos_list.append((single_idx, abs_diff, loss_diff))
                elif loss_diff < 0:
                    neg_list.append((single_idx, abs_diff, loss_diff))
        else:
            if loss_diff > 0:
                pos_list.append((idx, abs_diff, loss_diff))
            elif loss_diff < 0:
                neg_list.append((idx, abs_diff, loss_diff))

    # Sort by absolute loss difference (descending)
    pos_list.sort(key=lambda x: x[1], reverse=True)
    neg_list.sort(key=lambda x: x[1], reverse=True)

    # Extract sorted indices
    pos_indices = [item[0] for item in pos_list]
    neg_indices = [item[0] for item in neg_list]

    # Calculate statistics
    all_diffs_arr = np.array(all_diffs)

    # All Stats
    all_stats = {
        "Total_Keys": len(all_diffs_arr),
        "Total_Tokens": len(pos_list) + len(neg_list),
        "Min": float(np.min(all_diffs_arr)) if len(all_diffs_arr) > 0 else 0.0,
        "Max": float(np.max(all_diffs_arr)) if len(all_diffs_arr) > 0 else 0.0,
        "Mean": float(np.mean(all_diffs_arr)) if len(all_diffs_arr) > 0 else 0.0,
        "Std": float(np.std(all_diffs_arr)) if len(all_diffs_arr) > 0 else 0.0,
        "Median": float(np.median(all_diffs_arr)) if len(all_diffs_arr) > 0 else 0.0,
        "Q25": float(np.percentile(all_diffs_arr, 25)) if len(all_diffs_arr) > 0 else 0.0,
        "Q75": float(np.percentile(all_diffs_arr, 75)) if len(all_diffs_arr) > 0 else 0.0
    }

    # Positive Stats
    pos_diffs = np.array([item[2] for item in pos_list])
    pos_stats = {
        "Total": len(pos_diffs),
        "Total_Loss_Diff": float(np.sum(pos_diffs)) if len(pos_diffs) > 0 else 0.0,
        "Min": float(np.min(pos_diffs)) if len(pos_diffs) > 0 else 0.0,
        "Max": float(np.max(pos_diffs)) if len(pos_diffs) > 0 else 0.0,
        "Mean": float(np.mean(pos_diffs)) if len(pos_diffs) > 0 else 0.0,
        "Std": float(np.std(pos_diffs)) if len(pos_diffs) > 0 else 0.0,
        "Median": float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 0.0,
        "Q25": float(np.percentile(pos_diffs, 25)) if len(pos_diffs) > 0 else 0.0,
        "Q75": float(np.percentile(pos_diffs, 75)) if len(pos_diffs) > 0 else 0.0
    }

    # Negative Stats
    neg_diffs = np.array([item[2] for item in neg_list])
    neg_stats = {
        "Total": len(neg_diffs),
        "Total_Loss_Diff": float(np.sum(neg_diffs)) if len(neg_diffs) > 0 else 0.0,
        "Min": float(np.min(neg_diffs)) if len(neg_diffs) > 0 else 0.0,
        "Max": float(np.max(neg_diffs)) if len(neg_diffs) > 0 else 0.0,
        "Mean": float(np.mean(neg_diffs)) if len(neg_diffs) > 0 else 0.0,
        "Std": float(np.std(neg_diffs)) if len(neg_diffs) > 0 else 0.0,
        "Median": float(np.median(neg_diffs)) if len(neg_diffs) > 0 else 0.0,
        "Q25": float(np.percentile(neg_diffs, 25)) if len(neg_diffs) > 0 else 0.0,
        "Q75": float(np.percentile(neg_diffs, 75)) if len(neg_diffs) > 0 else 0.0
    }

    # Create output data
    pos_output = {
        "statistics": pos_stats,
        "tokens": [{"index": item[0], "loss_diff": item[2]} for item in pos_list]
    }

    neg_output = {
        "statistics": neg_stats,
        "tokens": [{"index": item[0], "loss_diff": item[2]} for item in neg_list]
    }

    # Save files
    with open(pos_file, 'w') as f:
        json.dump(pos_output, f, indent=2)

    with open(neg_file, 'w') as f:
        json.dump(neg_output, f, indent=2)

    # Print statistics
    print(f"Total Unique Loss Diffs: {len(data)}")
    print(f"Total Tokens (with duplicates): {all_stats['Total_Tokens']}")
    print(f"Positive tokens (loss increases when pruned): {pos_stats['Total']}")
    print(f"Negative tokens (loss decreases when pruned): {neg_stats['Total']}")

    print(f"\n--- All Loss Diffs Statistics ---")
    print(f"Total:  {all_stats['Total_Keys']}")
    print(f"Min:    {all_stats['Min']:.6f}")
    print(f"Max:    {all_stats['Max']:.6f}")
    print(f"Mean:   {all_stats['Mean']:.6f}")
    print(f"Std:    {all_stats['Std']:.6f}")
    print(f"Median: {all_stats['Median']:.6f}")
    print(f"Q25:    {all_stats['Q25']:.6f}")
    print(f"Q75:    {all_stats['Q75']:.6f}")

    print(f"\n--- Positive Loss Diffs Statistics ---")
    print(f"Total:              {pos_stats['Total']}")
    print(f"Total Loss Diff:    {pos_stats['Total_Loss_Diff']:.6f}")
    print(f"Min:                {pos_stats['Min']:.6f}")
    print(f"Max:                {pos_stats['Max']:.6f}")
    print(f"Mean:               {pos_stats['Mean']:.6f}")
    print(f"Std:                {pos_stats['Std']:.6f}")
    print(f"Median:             {pos_stats['Median']:.6f}")
    print(f"Q25:                {pos_stats['Q25']:.6f}")
    print(f"Q75:                {pos_stats['Q75']:.6f}")

    print(f"\n--- Negative Loss Diffs Statistics ---")
    print(f"Total:              {neg_stats['Total']}")
    print(f"Total Loss Diff:    {neg_stats['Total_Loss_Diff']:.6f}")
    print(f"Min:                {neg_stats['Min']:.6f}")
    print(f"Max:                {neg_stats['Max']:.6f}")
    print(f"Mean:               {neg_stats['Mean']:.6f}")
    print(f"Std:                {neg_stats['Std']:.6f}")
    print(f"Median:             {neg_stats['Median']:.6f}")
    print(f"Q25:                {neg_stats['Q25']:.6f}")
    print(f"Q75:                {neg_stats['Q75']:.6f}")

    print(f"\nPositive tokens saved to: {pos_file}")
    print(f"Negative tokens saved to: {neg_file}")

# Loop through all combinations
for layer in layer_indices:
    for sample in sample_indices:
        process_single_case(layer, sample)

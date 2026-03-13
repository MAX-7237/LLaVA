"""
绘图3: Top-K稳定性
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import load_sample_across_layers, compute_top_k_stability

def plot_topk_stability(stability, output_path):
    """绘制top-k稳定性变化"""
    plt.figure(figsize=(12, 6))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    markers = ['o', 's', '^', 'D']
    
    for idx, (k, vals) in enumerate(sorted(stability.items())):
        plt.plot(range(len(vals)), vals, marker=markers[idx], 
                label=f'Top-{k}', color=colors[idx], linewidth=2, markersize=6)
    
    plt.xlabel('Layer Transition (Layer i → Layer i+1)')
    plt.ylabel('Overlap Ratio')
    plt.title('Top-K Token Stability Across Layers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")

def main():
    base_path = '/data/users/Actor/LLaVA_Prune/results_lossing'
    output_dir = '/data/users/Actor/LLaVA_Prune/results_lossing/sample_analysis_output'
    
    for sample_idx in range(10):
        layer_data = load_sample_across_layers(base_path, sample_idx)
        if len(layer_data) == 0:
            continue
        
        stability = compute_top_k_stability(layer_data,top_k_values=[1])
        
        sample_output_dir = os.path.join(output_dir, f'sample_{sample_idx}')
        os.makedirs(sample_output_dir, exist_ok=True)
        
        plot_topk_stability(stability, 
            os.path.join(sample_output_dir, 'topk_stability.png'))

if __name__ == '__main__':
    main()

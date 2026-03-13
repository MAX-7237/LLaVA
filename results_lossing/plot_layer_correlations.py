"""
绘图2: 层间相关性变化
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import load_sample_across_layers, compute_layer_importance_correlation

def plot_layer_correlations(correlations, output_path):
    """绘制层间相关性变化"""
    pairs = sorted(correlations.keys())
    layer_pairs = [f"L{p[0]}->L{p[1]}" for p in pairs]
    spearman_corrs = [correlations[p]['spearman_corr'] for p in pairs]
    
    plt.figure(figsize=(14, 5))
    plt.plot(range(len(layer_pairs)), spearman_corrs, marker='o', color='green', linewidth=1.5)
    plt.fill_between(range(len(layer_pairs)), spearman_corrs, alpha=0.3, color='green')
    plt.xticks(range(len(layer_pairs)), layer_pairs, rotation=45, ha='right')
    plt.xlabel('Layer Transition')
    plt.ylabel('Spearman Correlation')
    plt.title('Token Importance Correlation Between Adjacent Layers')
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='0.5 baseline')
    plt.legend()
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
        
        correlations = compute_layer_importance_correlation(layer_data)
        
        sample_output_dir = os.path.join(output_dir, f'sample_{sample_idx}')
        os.makedirs(sample_output_dir, exist_ok=True)
        
        plot_layer_correlations(correlations, 
            os.path.join(sample_output_dir, 'layer_correlations.png'))

if __name__ == '__main__':
    main()

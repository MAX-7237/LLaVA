"""
绘图4: 均值和标准差演化
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import load_sample_across_layers, analyze_importance_distribution_shift

def plot_mean_std_evolution(analysis, output_path):
    """绘制均值和标准差随层数的变化"""
    sorted_layers = sorted(analysis.keys())
    means = [analysis[l]['mean'] for l in sorted_layers]
    stds = [analysis[l]['std'] for l in sorted_layers]
    maxs = [analysis[l]['max'] for l in sorted_layers]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(sorted_layers, means, marker='o', color='blue', linewidth=2)
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Mean Loss Diff')
    axes[0].set_title('Mean Importance Across Layers')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(sorted_layers, stds, marker='s', color='orange', linewidth=2)
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Std Loss Diff')
    axes[1].set_title('Std of Importance Across Layers')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(sorted_layers, maxs, marker='^', color='red', linewidth=2)
    axes[2].set_xlabel('Layer Index')
    axes[2].set_ylabel('Max Loss Diff')
    axes[2].set_title('Max (Top Token) Importance Across Layers')
    axes[2].grid(True, alpha=0.3)
    
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
        
        analysis = analyze_importance_distribution_shift(layer_data)
        
        sample_output_dir = os.path.join(output_dir, f'sample_{sample_idx}')
        os.makedirs(sample_output_dir, exist_ok=True)
        
        plot_mean_std_evolution(analysis, 
            os.path.join(sample_output_dir, 'mean_std_evolution.png'))

if __name__ == '__main__':
    main()

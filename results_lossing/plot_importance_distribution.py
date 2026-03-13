"""
绘图1: Token重要性分布 - 32层直方图
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import load_sample_across_layers

def plot_importance_across_layers(layer_data, sample_idx, output_path):
    """绘制同一样本在不同层的token重要性"""
    sorted_layers = sorted(layer_data.keys())
    n_layers = len(sorted_layers)
    
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, layer_idx in enumerate(sorted_layers):
        ax = axes[idx]
        loss_diffs = layer_data[layer_idx]['loss_diffs']
        
        ax.hist(loss_diffs, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(loss_diffs.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {loss_diffs.mean():.4f}')
        ax.set_title(f'Layer {layer_idx}', fontsize=9)
        ax.tick_params(axis='both', labelsize=7)
        
        if idx == 0:
            ax.set_ylabel('Frequency', fontsize=8)
        if idx == n_layers - 8:
            ax.set_xlabel('Loss Diff', fontsize=8)
    
    plt.suptitle(f'Sample {sample_idx}: Token Importance Distribution Across 32 Layers', fontsize=14)
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
        
        sample_output_dir = os.path.join(output_dir, f'sample_{sample_idx}')
        os.makedirs(sample_output_dir, exist_ok=True)
        
        plot_importance_across_layers(layer_data, sample_idx, 
            os.path.join(sample_output_dir, 'importance_distribution.png'))

if __name__ == '__main__':
    main()

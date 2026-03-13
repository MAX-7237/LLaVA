"""
绘图5: Top-K Token Index 分布变化
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import load_sample_across_layers, compute_importance_rankings

def plot_rank_consistency(layer_data, output_path, top_k=50):
    """
    绘制每一层Top-K Token的Index分布
    X轴：层
    Y轴：Token Index
    """
    rankings = compute_importance_rankings(layer_data)
    sorted_layers = sorted(layer_data.keys())
    
    top_k_indices = np.array([rankings[layer][:top_k] for layer in sorted_layers])
    
    plt.figure(figsize=(14, 7))
    
    for i in range(top_k):
        plt.plot(sorted_layers, top_k_indices[:, i], alpha=0.4, linewidth=1, color='steelblue')
    
    mean_indices = top_k_indices.mean(axis=1)
    plt.plot(sorted_layers, mean_indices, color='red', linewidth=3, label=f'Mean Index', marker='o')
    
    median_indices = np.median(top_k_indices, axis=1)
    plt.plot(sorted_layers, median_indices, color='orange', linewidth=2, label=f'Median Index', marker='s', linestyle='--')
    
    plt.xlabel('Layer Index')
    plt.ylabel('Token Index')
    plt.title(f'Top-{top_k} Token Index Distribution Across Layers')
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
        
        sample_output_dir = os.path.join(output_dir, f'sample_{sample_idx}')
        os.makedirs(sample_output_dir, exist_ok=True)
        
        for k in [5,20,50,100,200,300]:
            rank_consistency = os.path.join(sample_output_dir, f'rank_consistency_top{k}.png')
            plot_rank_consistency(layer_data, rank_consistency, top_k=k)

if __name__ == '__main__':
    main()

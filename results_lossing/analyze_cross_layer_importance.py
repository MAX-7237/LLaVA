"""
分析同一图像样本在32层LLM中Token重要性变化情况
"""
import numpy as np
import json
import os
import matplotlib.pyplot as plt

def load_sample_across_layers(base_path, sample_idx, max_layers=32):
    """加载同一样本在所有层的数据"""
    layer_data = {}
    
    for layer_idx in range(max_layers):
        layer_path = os.path.join(base_path, f'results_lossing_{layer_idx}', str(sample_idx))
        
        npz_path = os.path.join(layer_path, 'prune_token_loss_data.npz')
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            layer_data[layer_idx] = {
                'losses': data['all_losses'],
                'loss_diffs': data['loss_differences'],
                'baseline_loss': data['baseline_loss']
            }
    
    return layer_data

def compute_importance_rankings(layer_data):
    """计算每层token的重要性排名"""
    rankings = {}
    
    for layer_idx, data in layer_data.items():
        # 按loss差异降序排序，值越大表示token越重要
        sorted_indices = np.argsort(data['loss_diffs'])[::-1]
        rankings[layer_idx] = sorted_indices
    
    return rankings

def compute_layer_importance_correlation(layer_data):
    """计算相邻层之间token重要性的相关性（Spearman相关系数）"""
    from scipy import stats as scipy_stats
    
    correlations = {}
    sorted_layers = sorted(layer_data.keys())
    
    for i in range(len(sorted_layers) - 1):
        layer1 = sorted_layers[i]
        layer2 = sorted_layers[i + 1]
        
        # Spearman相关性
        corr, p_value = scipy_stats.spearmanr(
            layer_data[layer1]['loss_diffs'],
            layer_data[layer2]['loss_diffs']
        )
        
        correlations[(layer1, layer2)] = {
            'spearman_corr': corr,
            'p_value': p_value
        }
    
    return correlations

def compute_top_k_stability(layer_data, top_k_values=[10, 25, 50, 100]):
    """计算top-k token在层间的稳定性"""
    rankings = compute_importance_rankings(layer_data)
    sorted_layers = sorted(layer_data.keys())
    
    stability = {k: [] for k in top_k_values}
    
    for i in range(len(sorted_layers) - 1):
        layer1 = sorted_layers[i]
        layer2 = sorted_layers[i + 1]
        
        # 取每层的前k个token索引（排序后的）
        ranking1 = rankings[layer1]
        ranking2 = rankings[layer2]
        
        for k in top_k_values:
            # 前k个最重要的token索引
            top_tokens_l1 = set(ranking1[:k])
            top_tokens_l2 = set(ranking2[:k])
            
            overlap = len(top_tokens_l1 & top_tokens_l2) / k
            stability[k].append(overlap)
    
    return stability

def analyze_importance_distribution_shift(layer_data):
    """分析重要性分布在不同层之间的变化"""
    analysis = {}
    sorted_layers = sorted(layer_data.keys())
    
    # 第一层作为基准
    base_losses = layer_data[sorted_layers[0]]['loss_diffs']
    
    for layer_idx in sorted_layers:
        current_losses = layer_data[layer_idx]['loss_diffs']
        
        # 计算与基准的差异
        mean_diff = current_losses.mean() - base_losses.mean()
        std_ratio = current_losses.std() / base_losses.std()
        
        # 计算分布变化 (KL散度的近似)
        from scipy import stats as scipy_stats
        
        # 归一化后计算KL散度
        base_norm = (base_losses - base_losses.min()) / (base_losses.max() - base_losses.min() + 1e-8)
        curr_norm = (current_losses - current_losses.min()) / (current_losses.max() - current_losses.min() + 1e-8)
        
        analysis[layer_idx] = {
            'mean': current_losses.mean(),
            'std': current_losses.std(),
            'max': current_losses.max(),
            'mean_shift_from_base': mean_diff,
            'std_ratio_from_base': std_ratio
        }
    
    return analysis

def plot_importance_across_layers(layer_data, sample_idx, output_path):
    """绘制同一样本在不同层的token重要性"""
    sorted_layers = sorted(layer_data.keys())
    n_layers = len(sorted_layers)
    
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, layer_idx in enumerate(sorted_layers):
        ax = axes[idx]
        loss_diffs = layer_data[layer_idx]['loss_diffs']
        
        # 绘制分布直方图
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

def plot_rank_consistency(layer_data, output_path, top_k=50):
    """
    绘制每一层Top-K Token的Index分布
    X轴：层
    Y轴：Token Index
    """
    rankings = compute_importance_rankings(layer_data)
    sorted_layers = sorted(layer_data.keys())
    
    # 获取每层的Top-K token index
    # shape: (n_layers, top_k)
    top_k_indices = np.array([rankings[layer][:top_k] for layer in sorted_layers])
    
    plt.figure(figsize=(14, 7))
    
    # 画每一条线（每个token index在各层的变化）
    for i in range(top_k):
        plt.plot(sorted_layers, top_k_indices[:, i], alpha=0.4, linewidth=1, color='steelblue')
    
    # 画平均index
    mean_indices = top_k_indices.mean(axis=1)
    plt.plot(sorted_layers, mean_indices, color='red', linewidth=3, label=f'Mean Index', marker='o')
    
    # 画中位数
    median_indices = np.median(top_k_indices, axis=1)
    plt.plot(sorted_layers, median_indices, color='orange', linewidth=2, label=f'Median Index', marker='s', linestyle='--')
    
    plt.xlabel('Layer Index')
    plt.ylabel('Token Index')
    plt.title(f'Top-{top_k} Token Index Distribution Across Layers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 修改文件名，包含top_k
    base, ext = os.path.splitext(output_path)
    output_path_with_k = f"{base}_top{top_k}{ext}"
    plt.savefig(output_path_with_k, dpi=150)
    plt.close()
    print(f"Saved: {output_path_with_k}")

def main():
    base_path = '/data/users/Actor/LLaVA_Prune/results_lossing'
    output_dir = '/data/users/Actor/LLaVA_Prune/results_lossing/sample_analysis_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 分析样本
    for sample_idx in range(10):
        print(f"\n{'='*60}")
        print(f"Analyzing Sample {sample_idx}")
        print('='*60)
        
        layer_data = load_sample_across_layers(base_path, sample_idx)
        print(f"Loaded {len(layer_data)} layers for sample {sample_idx}")
        
        if len(layer_data) == 0:
            print(f"No data found for sample {sample_idx}")
            continue
        
        # 计算相关性
        print("\nComputing layer correlations...")
        correlations = compute_layer_importance_correlation(layer_data)
        
        # 计算稳定性
        print("Computing top-k stability...")
        stability = compute_top_k_stability(layer_data)
        
        # 分析分布变化
        print("Analyzing distribution shifts...")
        analysis = analyze_importance_distribution_shift(layer_data)
        
        # 打印关键统计
        print(f"\n{'Layer':<8} {'Mean':<12} {'Std':<12} {'Max':<12}")
        print("-"*44)
        for layer_idx in sorted(analysis.keys()):
            a = analysis[layer_idx]
            print(f"{layer_idx:<8} {a['mean']:<12.6f} {a['std']:<12.6f} {a['max']:<12.6f}")
        
        # 打印相关性
        print(f"\n{'Layer Pair':<15} {'Spearman Corr':<18} {'P-value':<15}")
        print("-"*48)
        for pair, info in correlations.items():
            print(f"{pair[0]}->{pair[1]:<12} {info['spearman_corr']:<18.4f} {info['p_value']:<15.2e}")
        
        # 绘图
        print("\nGenerating plots...")
        sample_output_dir = os.path.join(output_dir, f'sample_{sample_idx}')
        os.makedirs(sample_output_dir, exist_ok=True)
        
        plot_importance_across_layers(layer_data, sample_idx, 
            os.path.join(sample_output_dir, 'importance_distribution.png'))
        plot_layer_correlations(correlations, 
            os.path.join(sample_output_dir, 'layer_correlations.png'))
        plot_topk_stability(stability, 
            os.path.join(sample_output_dir, 'topk_stability.png'))
        plot_mean_std_evolution(analysis, 
            os.path.join(sample_output_dir, 'mean_std_evolution.png'))
        plot_rank_consistency(layer_data, 
            os.path.join(sample_output_dir, 'rank_consistency'), top_k=1)
        
        # 保存统计数据
        stats_output = {
            'distribution_analysis': analysis,
            'correlations': {f"{k[0]}->{k[1]}": v for k, v in correlations.items()},
            'stability': {f"top_{k}": v for k, v in stability.items()}
        }
        with open(os.path.join(sample_output_dir, 'statistics.json'), 'w') as f:
            json.dump(stats_output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Output saved to: {output_dir}")
    print('='*60)

if __name__ == '__main__':
    main()

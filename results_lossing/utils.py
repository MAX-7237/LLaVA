"""
共享工具函数：数据加载和基础计算
"""
import numpy as np
import os

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
        
        ranking1 = rankings[layer1]
        ranking2 = rankings[layer2]
        
        for k in top_k_values:
            top_tokens_l1 = set(ranking1[:k])
            top_tokens_l2 = set(ranking2[:k])
            
            overlap = len(top_tokens_l1 & top_tokens_l2) / k
            stability[k].append(overlap)
    
    return stability

def analyze_importance_distribution_shift(layer_data):
    """分析重要性分布在不同层之间的变化"""
    analysis = {}
    sorted_layers = sorted(layer_data.keys())
    
    base_losses = layer_data[sorted_layers[0]]['loss_diffs']
    
    for layer_idx in sorted_layers:
        current_losses = layer_data[layer_idx]['loss_diffs']
        
        mean_diff = current_losses.mean() - base_losses.mean()
        std_ratio = current_losses.std() / base_losses.std()
        
        analysis[layer_idx] = {
            'mean': current_losses.mean(),
            'std': current_losses.std(),
            'max': current_losses.max(),
            'mean_shift_from_base': mean_diff,
            'std_ratio_from_base': std_ratio
        }
    
    return analysis

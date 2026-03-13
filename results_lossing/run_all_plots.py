"""
打包所有绘图脚本
"""
import os
import subprocess

def main():
    scripts = [
        'plot_importance_distribution.py',
        'plot_layer_correlations.py', 
        'plot_topk_stability.py',
        'plot_mean_std_evolution.py',
        'plot_rank_consistency.py'
    ]
    
    for script in scripts:
        print(f"\nRunning {script}...")
        subprocess.run(['python', script])

if __name__ == '__main__':
    main()

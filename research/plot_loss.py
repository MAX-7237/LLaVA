import json
import matplotlib.pyplot as plt

k=576

# 循环范围
layer_range = [11,13,14,15,16,17,19,20,21,22,23,25,26,27,28,29,31,32]# 示例: 0 到 12
sample_range = range(0, 10) # 示例: 0

for layer_index in layer_range:
    for sample_index in sample_range:
        print(f"Processing layer {layer_index}, sample {sample_index}...")

        # Load the data
        with open(f"/data/users/Actor/LLaVA_Prune/results_lossing/results_lossing_{layer_index}/{sample_index}/loss_diff_sorted.json", "r") as f:
            data = json.load(f)

        # ... (rest of the plotting code) ...

        # Convert keys to floats for sorting
        items = [(float(k), v) for k, v in data.items()]

        # Sort by loss difference (descending)
        items.sort(key=lambda x: x[0], reverse=True)

        # Take top 100
        top_k = items[:k]

        # Extract x (index) and y (loss difference)
        # The structure is (loss_diff, token_id) or (loss_diff, list_of_token_ids)
        x = []
        y = []

        for loss_diff, token_id in top_k:
            if isinstance(token_id, list):
                # If there are multiple token_ids for the same loss_diff (e.g. line 536)
                # We treat each token_id individually
                for t_id in token_id:
                    x.append(t_id)
                    y.append(loss_diff)
            else:
                x.append(token_id)
                y.append(loss_diff)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(y)), y, marker='o') # X is just the order 0..49
        plt.xlabel('Rank (by Loss Difference)')
        plt.ylabel('Loss Difference')
        plt.title(f'Top {k} Tokens by Loss Difference')
        plt.grid(True)
        plt.savefig(f"results_lossing_{layer_index}/{sample_index}/top_{k}_loss_diff.png")
        print(f"Saved results_lossing_{layer_index}/{sample_index}/top_{k}_loss_diff.png")

        # Let's also provide the scatter plot of actual index vs loss just in case
        plt.figure(figsize=(10, 6))
        # x contains the token_ids
        plt.scatter(x, y, marker='o')
        plt.xlabel('Token Index')
        plt.ylabel('Loss Difference')
        plt.title(f'Top {k} Tokens: Loss Difference vs Token Index')
        plt.grid(True)
        plt.savefig(f"results_lossing_{layer_index}/{sample_index}/top_{k}_loss_diff_scatter.png")
        print(f"Saved results_lossing_{layer_index}/{sample_index}/top_{k}_loss_diff_scatter.png")

print("All done!")

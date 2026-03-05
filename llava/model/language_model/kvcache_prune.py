import torch
from transformers.cache_utils import Cache

def _cat_remove_1d(x: torch.Tensor, cut: int, dim: int):
    # remove index `cut` along `dim`
    return torch.cat([x.index_select(dim, torch.arange(0, cut, device=x.device)),
                      x.index_select(dim, torch.arange(cut + 1, x.size(dim), device=x.device))], dim=dim)


def _remove_index_along_dim(x: torch.Tensor, cut: int, dim: int):
    # faster slicing version (works for any dim)
    return torch.cat([x.narrow(dim, 0, cut), x.narrow(dim, cut + 1, x.size(dim) - cut - 1)], dim=dim)


def _prune_dynamic_cache_inplace(cache: Cache, cut: int):
    """
    In-place remove position `cut` from ALL cached K/V tensors in a HF Cache/DynamicCache object.
    Supports typical tensor layouts:
      - (B, H, S, D)  seq dim = -2
      - (B, S, H, D)  seq dim = -3
    """
    if cache is None:
        return

    # DynamicCache in HF usually has key_cache/value_cache lists
    if not (hasattr(cache, "key_cache") and hasattr(cache, "value_cache")):
        return  # unknown cache type, do nothing

    key_cache = cache.key_cache
    value_cache = cache.value_cache

    new_len = None

    for i in range(len(key_cache)):
        k = key_cache[i]
        v = value_cache[i]
        if k is None or v is None:
            continue

        # infer seq dim
        # Prefer -2 (B,H,S,D) if possible
        if k.dim() >= 3 and k.size(-2) > cut:
            seq_dim_k = -2
        elif k.dim() >= 4 and k.size(-3) > cut:
            seq_dim_k = -3
        else:
            # can't infer safely
            continue

        if v.dim() >= 3 and v.size(-2) > cut:
            seq_dim_v = -2
        elif v.dim() >= 4 and v.size(-3) > cut:
            seq_dim_v = -3
        else:
            continue

        key_cache[i] = _remove_index_along_dim(k, cut=cut, dim=seq_dim_k)
        value_cache[i] = _remove_index_along_dim(v, cut=cut, dim=seq_dim_v)

        if new_len is None:
            # after pruning, seq length is new size at seq dim
            new_len = key_cache[i].size(seq_dim_k)

    # keep seen_tokens consistent (HF uses either .seen_tokens or ._seen_tokens depending on version)
    if new_len is not None:
        if hasattr(cache, "seen_tokens"):
            cache.seen_tokens = new_len
        if hasattr(cache, "_seen_tokens"):
            cache._seen_tokens = new_len
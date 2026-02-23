
import torch
import torch.nn.functional as F

LOSS_REGISTRY = {} # way to store all functions of of loss that we use globally

def register_loss(name):
    def decorator(fn):
        LOSS_REGISTRY[name] = fn
        return fn
    return decorator

def get_loss_fn(name):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name]

@register_loss("mse")
def weighted_mse(pred, target, weights=None):
    if weights is None:
        return F.mse_loss(pred,target)
    return ((pred-target)**2 * weights).mean()

@register_loss("correlation_mse")
def correlation_mse_loss(pred, target, weights=None, alpha=0.5):
    if weights is not None:
        mse = ((pred-target) ** 2 * weights).mean()
    else:
        mse = F.mse_loss(pred, target)
    
    pred_c = pred - pred.mean()
    target_c = target - target.mean()
    pred_std = pred.std()
    target_std = target.std()

    if pred_std > 1e-8 and target_std > 1e-8:
        corr = (pred_c * target_c).mean() / (pred_std * target_std)
    else:
        corr = torch.tensor(0.0, device=pred.device)
    
    return mse + alpha * (1-corr)

@register_loss("top_k_focused")
def top_k_focussed_loss(pred, target, weights=None, k_frac = 0.2):
    base_mse = (pred-target)**2
    k = max(1, int(target.shape[-1] * k_frac))
    _, top_idx = torch.topk(target.abs(), k, dim=-1)
    importance = torch.ones_like(target)
    importance.scatter_(-1, top_idx, 5.0)

    if weights is not None:
        importance = importance * weights
    
    return (base_mse * importance).mean()

@register_loss("huber_corr")
def huber_correlation_loss(pred, target, weights=None, delta=0.1, alpha=0.5):
    huber = F.huber_loss(pred, target, delta=delta, reduction='none')

    if weights is not None:
        huber=huber*weights
    
    pred_c = pred - pred.mean()
    target_c = target + target.mean()
    pred_std = pred.std()
    target_std = target.std()

    if pred_std > 1e-8 and target_std > 1e-8:
        corr = (pred_c * target_c).mean() / (pred_std * target_std)
    else:
        corr = torch.tensor(0.0, device=pred.device)
    
    return huber.mean() + alpha * (1-corr)

@register_loss("autofocus_direction") # Proposed in GEARS, 2024
def autofocus_direction_loss(pred, target, weights=None, gamma=2, lambda_dir=1e-3, deg_threshold=0.1):
    deg_mask = target.abs() > deg_threshold

    if deg_mask.sum() == 0:
        deg_mask = torch.ones_like(target, dtype=torch.bool)

    per_gene_sq_err = (pred - target) ** 2

    if weights is not None:
        autofocus = ((per_gene_sq_err * weights)[deg_mask] ** gamma).mean()
    else:
        autofocus = (per_gene_sq_err[deg_mask] ** gamma).mean()

    pred_sign = torch.sign(pred[deg_mask])
    target_sign = torch.sign(target[deg_mask])
    dir_loss = ((pred_sign - target_sign) ** 2).mean()

    return autofocus + lambda_dir * dir_loss

@register_loss("deg_weighted_mse") # Loss proposed in "Diversity by Design", 2025
def deg_weighted_mse(pred, target, weights=None, deg_alpha=2.0):
    sq_error = (pred-target)**2

    if weights is not None:
        return (sq_error * weights).mean()
    else:
        de_proxy = target.abs() ** deg_alpha
        de_proxy = de_proxy / (de_proxy.mean() + 1e-8)
        return (sq_error * de_proxy).mean()
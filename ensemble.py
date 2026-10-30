"""
集成与校准工具 (所有超参数只在验证集上拟合, 测试集只用于最终报告, 避免数据泄漏)
- Temperature Scaling: 单模型概率校准
- ECE: 期望校准误差
- 集成搜索: 模型权重 + Max/Avg 混合系数 + 每类偏置 (替代原先手工设定的 cardboard/plastic x1.06)
"""
import itertools
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


# ==========================================
# 1. 校准
# ==========================================
def fit_temperature(logits, labels, max_iter=100):
    """在验证集 logits 上用 LBFGS 拟合单一温度 T (最小化 NLL)"""
    logits = logits.detach().float()
    labels = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0).item())


def expected_calibration_error(probs, labels, n_bins=15):
    probs = torch.as_tensor(probs, dtype=torch.float)
    labels = torch.as_tensor(labels, dtype=torch.long)
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.float().mean() * (conf[mask].mean() - correct[mask].mean()).abs()
    return float(ece.item())


# ==========================================
# 2. 集成融合
# ==========================================
def ensemble_probs(probs_list, weights=None, alpha=0.5, class_bias=None):
    """
    p = alpha * max_i(p_i) + (1 - alpha) * sum_i(w_i * p_i), 然后乘以每类偏置并归一化。
    兼容 torch.Tensor 输入, 可直接用于 ONNX 导出的计算图。
    """
    stacked = torch.stack(list(probs_list), dim=0)  # (M, B, C)
    m = stacked.shape[0]
    if weights is None:
        weights = torch.full((m,), 1.0 / m, dtype=stacked.dtype, device=stacked.device)
    weights = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device).view(m, 1, 1)
    p_avg = (stacked * weights).sum(dim=0) / weights.sum()
    p_max = stacked.max(dim=0).values
    p = alpha * p_max + (1.0 - alpha) * p_avg
    if class_bias is not None:
        p = p * torch.as_tensor(class_bias, dtype=p.dtype, device=p.device).view(1, -1)
    return p / p.sum(dim=1, keepdim=True)


def _macro_f1(probs, labels):
    return f1_score(labels, probs.argmax(dim=1).numpy(), average='macro')


def search_ensemble(probs_list, labels, weight_step=0.1, alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
                    bias_grid=(0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15), bias_rounds=2):
    """
    在验证集上搜索集成超参数 (目标: macro-F1, 平局时偏向更简单的配置)。
    1) 网格搜索模型权重 (单纯形, 步长 weight_step) 与 alpha
    2) 坐标上升搜索每类偏置 (粗网格 + 偏好 1.0, 降低在小验证集上过拟合的风险)
    """
    labels = np.asarray(labels)
    m = len(probs_list)
    n_cls = probs_list[0].shape[1]
    steps = int(round(1.0 / weight_step))

    best = {'score': -1.0, 'weights': [1.0 / m] * m, 'alpha': 0.5}
    for combo in itertools.product(range(steps + 1), repeat=m):
        if sum(combo) != steps:
            continue
        w = [c / steps for c in combo]
        for a in alphas:
            s = _macro_f1(ensemble_probs(probs_list, w, a), labels)
            if s > best['score'] + 1e-9:
                best = {'score': s, 'weights': w, 'alpha': a}

    bias = [1.0] * n_cls
    base = ensemble_probs(probs_list, best['weights'], best['alpha'])
    cur = _macro_f1(base, labels)
    for _ in range(bias_rounds):
        for c in range(n_cls):
            best_b, best_s = bias[c], cur
            for b in bias_grid:
                trial = list(bias)
                trial[c] = b
                s = _macro_f1(ensemble_probs(probs_list, best['weights'], best['alpha'], trial), labels)
                # 只有严格更优才替换; 平局时选更接近 1.0 的值
                if s > best_s + 1e-9 or (abs(s - best_s) <= 1e-9 and abs(b - 1.0) < abs(best_b - 1.0)):
                    best_b, best_s = b, s
            bias[c], cur = best_b, best_s

    return {'weights': best['weights'], 'alpha': best['alpha'], 'class_bias': bias, 'val_macro_f1': cur}


def save_ensemble_config(path, config):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_ensemble_config(path, num_models=3, num_classes=6):
    """读取验证集拟合的集成配置; 不存在时退回旧版手工默认值"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        bias = [1.0] * num_classes
        bias[0], bias[4] = 1.06, 1.06
        return {'weights': [1.0 / num_models] * num_models, 'alpha': 0.5, 'class_bias': bias,
                'temperatures': [1.0] * num_models, 'img_size': 224}

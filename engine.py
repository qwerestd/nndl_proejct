"""
训练 / 评估 / 效率测试引擎 (train_evaluate.py 与 ablation.py 共用)
P1 改进:
  - logits + CrossEntropyLoss(label_smoothing) / 可选 Focal Loss
  - Mixup / CutMix
  - AdamW (weight_decay=0.05, norm/bias/layer_scale 不衰减) + 差分学习率 + 线性 warmup + 逐 step 余弦退火
  - 梯度裁剪, 模型 EMA
  - 以验证集 macro-F1 选择最优 checkpoint (平局看 val loss)
  - 冻结浅层时同时冻结其 BN 统计量
"""
import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score


# ==========================================
# 通用工具
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    epochs: int = 30
    warmup_epochs: int = 3
    backbone_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.05
    min_lr_ratio: float = 0.01
    label_smoothing: float = 0.1
    loss: str = 'ce'  # 'ce' | 'focal'
    focal_gamma: float = 2.0
    mix_prob: float = 0.5  # 每个 batch 使用 Mixup/CutMix 的概率 (0 关闭)
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    cutmix_switch_prob: float = 0.5  # 混合时选择 CutMix 的概率
    grad_clip: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.998
    patience: int = 10
    freeze_early: bool = False
    aux_weights: tuple = (0.5, 0.25)  # 与模型返回的辅助输出顺序一一对应
    freeze_keys: tuple = ('dense_stage1', 'mobile_stage1')
    backbone_keys: tuple = ('branch1_features', 'branch2_features', 'swin_norm', 'dense_stage', 'mobile_stage')
    extra: dict = field(default_factory=dict)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.register_buffer('weight', weight)
        self.gamma, self.ls = gamma, label_smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=1)
        ce = F.nll_loss(logp, target, weight=self.weight, reduction='none')
        if self.ls > 0:
            smooth = -logp.mean(dim=1)
            ce = (1 - self.ls) * ce + self.ls * smooth
        pt = logp.gather(1, target[:, None]).squeeze(1).exp()
        return ((1 - pt) ** self.gamma * ce).mean()


def build_criterion(cfg, class_weights, device):
    w = class_weights.to(device) if class_weights is not None else None
    if cfg.loss == 'focal':
        return FocalLoss(w, cfg.focal_gamma, cfg.label_smoothing).to(device)
    return nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)


# ==========================================
# Mixup / CutMix
# ==========================================
def mix_batch(inputs, labels, cfg):
    """返回 (混合后的输入, y_a, y_b, lam); 未触发时 lam=1"""
    if cfg.mix_prob <= 0 or random.random() > cfg.mix_prob:
        return inputs, labels, labels, 1.0
    perm = torch.randperm(inputs.size(0), device=inputs.device)
    if random.random() < cfg.cutmix_switch_prob:
        lam = np.random.beta(cfg.cutmix_alpha, cfg.cutmix_alpha)
        h, w = inputs.shape[2:]
        cut_h, cut_w = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)
        x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
        inputs = inputs.clone()
        inputs[:, :, y1:y2, x1:x2] = inputs[perm, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)  # 按实际裁剪面积修正
    else:
        lam = np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha)
        inputs = lam * inputs + (1 - lam) * inputs[perm]
    return inputs, labels, labels[perm], float(lam)


# ==========================================
# EMA
# ==========================================
class ModelEMA:
    def __init__(self, model, decay=0.998):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))  # 训练初期快速跟随
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


# ==========================================
# 优化器 & 学习率调度
# ==========================================
# 1-D 参数 (Norm 权重 / bias) 自动不衰减; 以下为额外不衰减的高维参数 (ConvNeXt layer_scale, Swin 相对位置偏置, GRN)
_NO_DECAY_KEYS = ('layer_scale', 'relative_position_bias_table', '.gamma', '.beta')


def build_optimizer(model, cfg):
    groups = {(b, d): [] for b in (True, False) for d in (True, False)}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = n.startswith(cfg.backbone_keys)
        no_decay = p.ndim <= 1 or n.endswith('.bias') or any(k in n for k in _NO_DECAY_KEYS)
        groups[(is_backbone, not no_decay)].append(p)

    param_groups = []
    for (is_backbone, decay), params in groups.items():
        if params:
            param_groups.append({'params': params,
                                 'lr': cfg.backbone_lr if is_backbone else cfg.head_lr,
                                 'weight_decay': cfg.weight_decay if decay else 0.0})
    return torch.optim.AdamW(param_groups)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return 0.01 + 0.99 * step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _apply_freeze(model, cfg):
    frozen = []
    if cfg.freeze_early:
        for name, module in model.named_children():
            if name in cfg.freeze_keys:
                for p in module.parameters():
                    p.requires_grad = False
                frozen.append(module)
    return frozen


def _split_outputs(outs):
    if isinstance(outs, (tuple, list)):
        return outs[0], list(outs[1:])
    return outs, []


# ==========================================
# 评估
# ==========================================
@torch.no_grad()
def evaluate(model, loader, device, use_amp=True):
    """返回 dict(loss, acc, macro_f1); loss 为不加权、不平滑的标准 CE, 便于跨配置比较"""
    model.eval()
    total_loss, preds, labels_all = 0.0, [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=use_amp and device.type == 'cuda'):
            logits = model(inputs)
        total_loss += F.cross_entropy(logits.float(), labels, reduction='sum').item()
        preds.append(logits.argmax(1).cpu())
        labels_all.append(labels.cpu())
    preds, labels_all = torch.cat(preds).numpy(), torch.cat(labels_all).numpy()
    n = len(labels_all)
    return {'loss': total_loss / n, 'acc': float((preds == labels_all).mean() * 100),
            'macro_f1': float(f1_score(labels_all, preds, average='macro') * 100)}


@torch.no_grad()
def collect_logits(model, loader, device, tta=True, use_amp=True):
    """收集 logits (TTA: 原图 + 水平翻转的 logits 平均), 用于温度校准与集成"""
    model.eval()
    all_logits, all_labels = [], []
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=use_amp and device.type == 'cuda'):
            logits = model(inputs).float()
            if tta:
                logits = (logits + model(torch.flip(inputs, dims=[3])).float()) / 2.0
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels).numpy()


# ==========================================
# 训练
# ==========================================
def train_single_model(model, model_name, train_loader, valid_loader, class_weights, save_path, cfg, device):
    print(f"\n{'=' * 60}\nStart Training [{model_name}] on {device}\n{'=' * 60}")
    model = model.to(device)
    frozen_modules = _apply_freeze(model, cfg)
    criterion = build_criterion(cfg, class_weights, device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    ema = ModelEMA(model, cfg.ema_decay) if cfg.use_ema else None

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best = {'macro_f1': -1.0, 'loss': float('inf')}
    history, patience = [], 0

    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        for m in frozen_modules:
            m.eval()  # 冻结层的 BN 统计量也保持不变
        train_loss, train_correct, seen = 0.0, 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            inputs, y_a, y_b, lam = mix_batch(inputs, labels, cfg)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device.type, enabled=use_amp):
                out_main, aux_outs = _split_outputs(model(inputs))
                outs = [out_main] + aux_outs
                ws = [1.0] + list(cfg.aux_weights[:len(aux_outs)])
                loss = sum(w * (lam * criterion(o, y_a) + (1 - lam) * criterion(o, y_b)) for w, o in zip(ws, outs))

            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            pred = out_main.argmax(1)
            train_loss += loss.item() * inputs.size(0)
            train_correct += (lam * (pred == y_a).float() + (1 - lam) * (pred == y_b).float()).sum().item()
            seen += inputs.size(0)

        # 验证: 原模型与 EMA 模型都评估, 取更优者
        candidates = {'raw': (model, evaluate(model, valid_loader, device))}
        if ema is not None:
            candidates['ema'] = (ema.module, evaluate(ema.module, valid_loader, device))
        tag, (cand_model, metrics) = max(candidates.items(),
                                         key=lambda kv: (kv[1][1]['macro_f1'], -kv[1][1]['loss']))

        log = {'epoch': epoch + 1, 'train_loss': train_loss / seen, 'train_acc': train_correct / seen * 100,
               'lr_backbone': optimizer.param_groups[0]['lr'],
               **{f'{k}_{m}': v[1][m] for k, v in candidates.items() for m in ('loss', 'acc', 'macro_f1')}}
        history.append(log)
        ema_str = (f" | EMA F1: {candidates['ema'][1]['macro_f1']:.2f}%" if 'ema' in candidates else '')
        print(f"Epoch [{epoch + 1:02d}/{cfg.epochs:02d}] ({time.time() - t0:.1f}s) | "
              f"Train Loss: {log['train_loss']:.4f}, Acc: {log['train_acc']:.2f}% | "
              f"Valid Loss: {candidates['raw'][1]['loss']:.4f}, Acc: {candidates['raw'][1]['acc']:.2f}%, "
              f"F1: {candidates['raw'][1]['macro_f1']:.2f}%{ema_str}")

        if (metrics['macro_f1'], -metrics['loss']) > (best['macro_f1'], -best['loss']):
            best = {**metrics, 'epoch': epoch + 1, 'source': tag}
            torch.save(cand_model.state_dict(), save_path)
            print(f"  --> Best [{model_name}] saved from {tag} (Valid F1: {metrics['macro_f1']:.2f}%, "
                  f"Acc: {metrics['acc']:.2f}%)")
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    # 保存训练配置与曲线, 便于复现与写报告
    meta = {'model_name': model_name, 'best': best, 'config': asdict(cfg), 'history': history}
    with open(os.path.splitext(save_path)[0] + '_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, default=str)
    return best


# ==========================================
# 效率测试: 参数量 / FLOPs / 延迟
# ==========================================
@torch.no_grad()
def benchmark_model(model, img_size=224, device=torch.device('cpu'), warmup=5, runs=20):
    model = model.eval().to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    x = torch.randn(1, 3, img_size, img_size, device=device)

    gflops = float('nan')
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            model(x)
        gflops = fc.get_total_flops() / 1e9
    except Exception as e:  # noqa: BLE001
        print(f"[benchmark] FLOPs counting failed: {e}")

    for _ in range(warmup):
        model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) / runs * 1000
    return {'params_m': params, 'gflops': gflops, 'latency_ms': latency_ms, 'device': device.type}

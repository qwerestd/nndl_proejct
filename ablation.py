"""
ModernHybridNet 消融实验 (P2)
用法示例:
    python ablation.py                                  # 跑全部变体, seed=42
    python ablation.py --variants full,no_grn --seeds 42,43,44
    python ablation.py --epochs 20 --retrain
输出:
    outputs/ablation/<variant>_s<seed>.pt (+ _meta.json)
    outputs/ablation/ablation_results.csv     每个 (变体, seed) 一行
    outputs/ablation/ablation_summary.md      mean ± std 汇总表 (可直接贴进报告)
"""
import argparse
import csv
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from dataset import prepare_dataloaders
from engine import TrainConfig, set_seed, train_single_model, collect_logits, benchmark_model
from models import ModernHybridNet

# 变体名 -> (描述, ModernHybridNet 构造参数)
VARIANTS = {
    'full': ('ConvNeXt+Swin, concat, SE, GRN, aux (完整模型)', {}),
    'convnext_only': ('仅 ConvNeXt-T 分支', dict(branches='convnext')),
    'swin_only': ('仅 Swin-T 分支', dict(branches='swin')),
    'no_grn': ('去掉 GRN', dict(use_grn=False)),
    'no_se': ('去掉 SE (分支 SE + 融合后 SE)', dict(use_se=False)),
    'no_aux': ('去掉辅助头 (无深监督)', dict(use_aux=False)),
    'gated': ('门控融合替代拼接', dict(fusion='gated')),
    'cross_attn': ('双向交叉注意力融合替代拼接', dict(fusion='cross_attn')),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--variants', default=','.join(VARIANTS))
    p.add_argument('--seeds', default='42', help='逗号分隔, 多个 seed 用于报告 mean±std')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--no-tta', action='store_true')
    p.add_argument('--retrain', action='store_true', help='忽略已有 checkpoint 重新训练')
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'dataset-resized')
    out_dir = os.path.join(base_dir, 'outputs', 'ablation')
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"未知变体 {unknown}, 可选: {list(VARIANTS)}")
    seeds = [int(s) for s in args.seeds.split(',')]

    # 数据划分固定 (split_seed=42), 训练 seed 只影响初始化 / 增强 / 采样顺序
    train_loader, valid_loader, test_loader, class_weights, class_names, _ = prepare_dataloaders(
        data_dir, batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers)
    num_classes = len(class_names)

    rows = []
    for variant in variants:
        desc, kwargs = VARIANTS[variant]
        eff = None
        for seed in seeds:
            set_seed(seed)
            ckpt = os.path.join(out_dir, f'{variant}_s{seed}.pt')
            model = ModernHybridNet(num_classes=num_classes, **kwargs).to(device)
            if args.retrain or not os.path.exists(ckpt):
                cfg = TrainConfig(epochs=args.epochs)
                train_single_model(model, f'Ablation-{variant}-s{seed}', train_loader, valid_loader,
                                   class_weights, ckpt, cfg, device)
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

            logits, labels = collect_logits(model, test_loader, device, tta=not args.no_tta)
            preds = logits.argmax(1).numpy()
            if eff is None:  # 效率指标与 seed 无关, 只测一次
                eff = benchmark_model(model, args.img_size, torch.device('cpu'))
            row = {'variant': variant, 'description': desc, 'seed': seed,
                   'test_acc': accuracy_score(labels, preds) * 100,
                   'test_macro_f1': f1_score(labels, preds, average='macro') * 100,
                   'params_m': eff['params_m'], 'gflops': eff['gflops'], 'cpu_latency_ms': eff['latency_ms']}
            rows.append(row)
            print(f"[{variant} | seed {seed}] Acc {row['test_acc']:.2f}% | Macro-F1 {row['test_macro_f1']:.2f}%")
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    with open(os.path.join(out_dir, 'ablation_results.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # mean ± std 汇总 (Markdown)
    lines = ['| Variant | Description | Acc (%) | Macro-F1 (%) | Params (M) | GFLOPs | CPU Latency (ms) |',
             '|---|---|---|---|---|---|---|']
    for variant in variants:
        rs = [r for r in rows if r['variant'] == variant]
        acc, f1 = np.array([r['test_acc'] for r in rs]), np.array([r['test_macro_f1'] for r in rs])
        std = (lambda a: f" ± {a.std(ddof=1):.2f}" if len(a) > 1 else '')
        lines.append(f"| {variant} | {rs[0]['description']} | {acc.mean():.2f}{std(acc)} | {f1.mean():.2f}{std(f1)} | "
                     f"{rs[0]['params_m']:.2f} | {rs[0]['gflops']:.2f} | {rs[0]['cpu_latency_ms']:.1f} |")
    table = '\n'.join(lines)
    with open(os.path.join(out_dir, 'ablation_summary.md'), 'w', encoding='utf-8') as f:
        f.write(f"# ModernHybridNet Ablation (seeds={seeds}, img_size={args.img_size})\n\n{table}\n")
    print('\n' + table)


if __name__ == '__main__':
    main()

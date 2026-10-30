import argparse
import csv
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from dataset import prepare_dataloaders
from engine import TrainConfig, set_seed, train_single_model, collect_logits, benchmark_model
from ensemble import (fit_temperature, expected_calibration_error, ensemble_probs, search_ensemble,
                      save_ensemble_config)
from models import HybridTrashNet, RWCNet, ModernHybridNet, MobileDeployWrapper


def get_model_specs(args):
    modern_suffix = '' if args.modern_fusion == 'concat' else f'_{args.modern_fusion}'
    return {
        # 1. SOTA Baseline 1: Hybrid-ResSeMo (论文 1)
        'hybrid': dict(label='Hybrid-ResSeMo', short='ResSeMo', build=lambda n: HybridTrashNet(num_classes=n),
                       ckpt='hybrid_ressemo_best.pt', cm='cm_hybrid.png', cmap='Blues',
                       cfg=dict(epochs=args.epochs or 30)),
        # 2. SOTA Baseline 2: RWC-Net (论文 2, 开启浅层冻结加速)
        'rwc': dict(label='RWC-Net', short='RWC-Net', build=lambda n: RWCNet(num_classes=n),
                    ckpt='rwcnet_best.pt', cm='cm_rwcnet.png', cmap='Greens',
                    cfg=dict(epochs=args.epochs or 20, freeze_early=True)),
        # 3. Proposed Novel SOTA: ModernHybridNet (ConvNeXt + Swin-T + GRN)
        'modern': dict(label='Modern-ConvNeXt-Swin', short='ModernHybrid',
                       build=lambda n: ModernHybridNet(num_classes=n, fusion=args.modern_fusion),
                       ckpt=f'modern_convnext_swin{modern_suffix}_best.pt', cm='cm_modern.png', cmap='Oranges',
                       cfg=dict(epochs=args.epochs or 30)),
    }


def make_train_config(args, overrides):
    return TrainConfig(warmup_epochs=args.warmup_epochs, weight_decay=args.weight_decay,
                       label_smoothing=args.label_smoothing, loss=args.loss, mix_prob=args.mix_prob,
                       use_ema=not args.no_ema, **overrides)


def save_confusion_matrix(labels, preds, class_names, title, save_path, cmap='Blues'):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_efficiency(rows, save_path):
    """精度-计算量权衡图 (气泡大小 = 参数量)"""
    plt.figure(figsize=(7, 5))
    for r in rows:
        plt.scatter(r['gflops'], r['test_acc'], s=max(r['params_m'], 1) * 6, alpha=0.6)
        plt.annotate(f"{r['name']}\n{r['params_m']:.1f}M", (r['gflops'], r['test_acc']),
                     textcoords='offset points', xytext=(6, 6), fontsize=9)
    plt.xlabel('GFLOPs (224x224, batch=1)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Accuracy vs. Compute (bubble size = #Params)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description='TrashNet training / calibrated ensemble / efficiency benchmark')
    p.add_argument('--img-size', type=int, default=224, help='输入分辨率, 可尝试 288 / 320')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=None, help='覆盖所有模型的训练轮数')
    p.add_argument('--warmup-epochs', type=int, default=3)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--loss', choices=['ce', 'focal'], default='ce')
    p.add_argument('--mix-prob', type=float, default=0.5, help='Mixup/CutMix 触发概率, 0 关闭')
    p.add_argument('--no-ema', action='store_true')
    p.add_argument('--modern-fusion', choices=['concat', 'gated', 'cross_attn'], default='concat')
    p.add_argument('--retrain', default='', help="重新训练的模型: 'all' 或逗号分隔 hybrid,rwc,modern")
    p.add_argument('--no-tta', action='store_true')
    p.add_argument('--skip-benchmark', action='store_true')
    p.add_argument('--skip-export', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'dataset-resized')
    out_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, test_loader, class_weights, class_names, _ = prepare_dataloaders(
        data_dir, batch_size=args.batch_size, img_size=args.img_size, num_workers=args.num_workers)
    num_classes = len(class_names)

    specs = get_model_specs(args)
    retrain = set(specs) if args.retrain == 'all' else {s.strip() for s in args.retrain.split(',') if s.strip()}

    # ==========================================================
    # 1-3. 训练或加载三个模型
    # ==========================================================
    loaded = {}
    for key, spec in specs.items():
        ckpt = os.path.join(out_dir, spec['ckpt'])
        model = spec['build'](num_classes).to(device)
        if key in retrain or not os.path.exists(ckpt):
            cfg = make_train_config(args, spec['cfg'])
            train_single_model(model, spec['label'], train_loader, valid_loader, class_weights, ckpt, cfg, device)
        else:
            print(f"[{spec['label']}] 加载已有 checkpoint: {ckpt} (如需使用新训练策略请加 --retrain {key})")
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        loaded[key] = model.eval()

    # ==========================================================
    # 4. 验证集拟合: 温度校准 + 集成超参数 (测试集完全不参与调参)
    # ==========================================================
    tta = not args.no_tta
    val_logits, test_logits, temps = {}, {}, {}
    for key, model in loaded.items():
        val_logits[key], val_labels = collect_logits(model, valid_loader, device, tta=tta)
        test_logits[key], test_labels = collect_logits(model, test_loader, device, tta=tta)
        temps[key] = fit_temperature(val_logits[key], val_labels)

    keys = list(loaded)
    val_probs = [F.softmax(val_logits[k] / temps[k], dim=1) for k in keys]
    test_probs = {k: F.softmax(test_logits[k] / temps[k], dim=1) for k in keys}

    ens_cfg = search_ensemble(val_probs, val_labels)
    ens_cfg.update({'models': keys, 'temperatures': [temps[k] for k in keys], 'img_size': args.img_size,
                    'tta': tta, 'checkpoints': [specs[k]['ckpt'] for k in keys]})
    save_ensemble_config(os.path.join(out_dir, 'ensemble_config.json'), ens_cfg)
    print("\n[Ensemble] 验证集拟合结果: "
          f"weights={[round(w, 2) for w in ens_cfg['weights']]}, alpha={ens_cfg['alpha']}, "
          f"class_bias={dict(zip(class_names, ens_cfg['class_bias']))}, val macro-F1={ens_cfg['val_macro_f1']:.4f}")

    probs_ensemble = ensemble_probs([test_probs[k] for k in keys], ens_cfg['weights'], ens_cfg['alpha'],
                                    ens_cfg['class_bias'])

    # ==========================================================
    # 5. 独立测试集评估 (只运行一次, 直接用于 Project Report)
    # ==========================================================
    eval_items = [(f"{i + 1}. [{specs[k]['label']}]", specs[k]['short'], test_probs[k],
                   F.softmax(test_logits[k], dim=1), f"{specs[k]['label']} Confusion Matrix",
                   specs[k]['cm'], specs[k]['cmap']) for i, k in enumerate(keys)]
    eval_items.append((f"{len(keys) + 1}. [Proposed Calibrated Ensemble]", 'Calib-Ensemble', probs_ensemble, None,
                       "Calibrated Ensemble Confusion Matrix", 'cm_ensemble.png', 'Purples'))

    summary = {}
    for header, short, probs, raw_probs, cm_title, cm_file, cmap in eval_items:
        preds = probs.argmax(dim=1).numpy()
        acc = accuracy_score(test_labels, preds) * 100
        mf1 = f1_score(test_labels, preds, average='macro') * 100
        ece = expected_calibration_error(probs, test_labels) * 100
        ece_raw = expected_calibration_error(raw_probs, test_labels) * 100 if raw_probs is not None else None
        summary[short] = {'test_acc': acc, 'test_macro_f1': mf1, 'ece': ece, 'ece_before_ts': ece_raw}

        print("\n" + "=" * 60)
        ece_str = f"ECE: {ece:.2f}%" + (f" (TS 前 {ece_raw:.2f}%)" if ece_raw is not None else '')
        print(f"{header} Test Accuracy: {acc:.2f}% | Macro-F1: {mf1:.2f}% | {ece_str}")
        print(classification_report(test_labels, preds, target_names=class_names, digits=4))
        save_confusion_matrix(test_labels, preds, class_names, cm_title, os.path.join(out_dir, cm_file), cmap)

    # ==========================================================
    # 6. 效率对比: Params / GFLOPs / 延迟, 与精度画在同一张图中
    # ==========================================================
    if not args.skip_benchmark:
        rows = []
        bench_device = torch.device('cpu')  # 以 CPU 单图延迟近似移动端场景
        for k in keys:
            b = benchmark_model(loaded[k], args.img_size, bench_device)
            rows.append({'name': specs[k]['short'], **b, **summary[specs[k]['short']]})
            loaded[k].to(device)
        rows.append({'name': 'Calib-Ensemble', 'device': 'cpu',
                     'params_m': sum(r['params_m'] for r in rows), 'gflops': sum(r['gflops'] for r in rows),
                     'latency_ms': sum(r['latency_ms'] for r in rows), **summary['Calib-Ensemble']})

        print("\n" + "=" * 92)
        print(f"{'Model':<16} | {'Params(M)':>9} | {'GFLOPs':>7} | {'CPU Lat(ms)':>11} | "
              f"{'Acc(%)':>7} | {'MacroF1(%)':>10} | {'ECE(%)':>6}")
        print("-" * 92)
        for r in rows:
            print(f"{r['name']:<16} | {r['params_m']:>9.2f} | {r['gflops']:>7.2f} | {r['latency_ms']:>11.1f} | "
                  f"{r['test_acc']:>7.2f} | {r['test_macro_f1']:>10.2f} | {r['ece']:>6.2f}")
        print("=" * 92)

        with open(os.path.join(out_dir, 'results_summary.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        plot_efficiency(rows, os.path.join(out_dir, 'accuracy_vs_compute.png'))

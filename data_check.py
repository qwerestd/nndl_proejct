"""
数据泄漏检查 (P2): 用感知哈希 (pHash) 查找近重复图片, 并统计跨 train/valid/test 的重复对。
仅依赖 PIL + numpy。
用法:
    python data_check.py                 # 默认汉明距离阈值 6 (64-bit pHash)
    python data_check.py --threshold 4 --plot
输出:
    outputs/data_check/near_duplicates.csv
    outputs/data_check/cross_split_pairs.png (--plot)
"""
import argparse
import csv
import os
from collections import Counter

import numpy as np
from PIL import Image
from torchvision import datasets

from dataset import split_indices


def _dct_matrix(n):
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2.0)
    return m


_DCT32 = _dct_matrix(32)


def phash(path, hash_size=8):
    """经典 pHash: 32x32 灰度 -> 2D DCT -> 取左上 8x8 低频 (去掉 DC) 与中位数比较"""
    img = np.asarray(Image.open(path).convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    dct = _DCT32 @ img @ _DCT32.T
    low = dct[:hash_size, :hash_size].flatten()
    return (low > np.median(low[1:])).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--threshold', type=int, default=6, help='汉明距离 <= 阈值视为近重复 (0 = 完全相同)')
    p.add_argument('--plot', action='store_true', help='可视化前 12 对跨划分重复样本')
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'dataset-resized')
    out_dir = os.path.join(base_dir, 'outputs', 'data_check')
    os.makedirs(out_dir, exist_ok=True)

    ds = datasets.ImageFolder(root=data_dir)
    paths = [s[0] for s in ds.samples]
    targets = np.array(ds.targets)
    train_idx, valid_idx, test_idx = split_indices(targets, seed=42)
    split_of = np.empty(len(paths), dtype=object)
    split_of[train_idx], split_of[valid_idx], split_of[test_idx] = 'train', 'valid', 'test'

    print(f"Computing pHash for {len(paths)} images ...")
    hashes = np.stack([phash(pth) for pth in paths]).astype(np.float32)  # (N, 64)
    # 汉明距离矩阵: d(i,j) = sum(h_i != h_j)
    dist = hashes @ (1 - hashes).T + (1 - hashes) @ hashes.T
    iu, ju = np.triu_indices(len(paths), k=1)
    mask = dist[iu, ju] <= args.threshold
    pairs = sorted(zip(iu[mask], ju[mask], dist[iu, ju][mask].astype(int)), key=lambda t: t[2])

    rows, kinds = [], Counter()
    for i, j, d in pairs:
        kind = '-'.join(sorted([split_of[i], split_of[j]]))
        cross = split_of[i] != split_of[j]
        kinds[kind] += 1
        rows.append({'hamming': d, 'path_a': os.path.relpath(paths[i], data_dir),
                     'path_b': os.path.relpath(paths[j], data_dir), 'split_a': split_of[i], 'split_b': split_of[j],
                     'class_a': ds.classes[targets[i]], 'class_b': ds.classes[targets[j]],
                     'cross_split': cross, 'label_conflict': targets[i] != targets[j]})

    csv_path = os.path.join(out_dir, 'near_duplicates.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fields = ['hamming', 'path_a', 'path_b', 'split_a', 'split_b', 'class_a', 'class_b', 'cross_split',
                  'label_conflict']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    cross_rows = [r for r in rows if r['cross_split']]
    test_leak = {r['path_a'] if r['split_a'] == 'test' else r['path_b'] for r in cross_rows
                 if 'test' in (r['split_a'], r['split_b']) and 'train' in (r['split_a'], r['split_b'])}
    print(f"\n近重复对 (hamming <= {args.threshold}): {len(rows)}")
    for k, v in sorted(kinds.items()):
        print(f"  {k:<12}: {v}")
    print(f"跨划分重复对: {len(cross_rows)} | 与训练集重复的测试图片: {len(test_leak)} / {len(test_idx)} "
          f"({len(test_leak) / len(test_idx) * 100:.2f}%)")
    print(f"标签冲突对 (近重复但类别不同): {sum(r['label_conflict'] for r in rows)}")
    print(f"详细列表: {csv_path}")

    if args.plot and cross_rows:
        import matplotlib.pyplot as plt
        show = cross_rows[:12]
        fig, axes = plt.subplots(len(show), 2, figsize=(6, 2.6 * len(show)), squeeze=False)
        for r_i, r in enumerate(show):
            for c_i, key in enumerate(('a', 'b')):
                axes[r_i, c_i].imshow(Image.open(os.path.join(data_dir, r[f'path_{key}'])).convert('RGB'))
                axes[r_i, c_i].set_title(f"{r[f'split_{key}']} | {r[f'class_{key}']}\n"
                                         f"{os.path.basename(r[f'path_{key}'])} (d={r['hamming']})", fontsize=8)
                axes[r_i, c_i].axis('off')
        plt.tight_layout()
        fig_path = os.path.join(out_dir, 'cross_split_pairs.png')
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"可视化: {fig_path}")


if __name__ == '__main__':
    main()

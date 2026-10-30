import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size=224, strong_aug=True):
    """
    构建 train / valid / test 预处理。
    - 训练: 直接 RandomResizedCrop 到目标尺寸 (修复原先 RandomResizedCrop(256)+CenterCrop(224) 抵消缩放增强的问题)
            + TrivialAugmentWide (自动增强) + RandomErasing (遮挡鲁棒)
    - 验证/测试: 按 256/224 比例 Resize 后中心裁剪
    """
    resize_size = int(round(img_size * 256 / 224))
    train_ops = [
        transforms.RandomResizedCrop(size=img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if strong_aug:
        train_ops.append(transforms.TrivialAugmentWide())
    else:
        train_ops += [transforms.RandomRotation(degrees=15), transforms.ColorJitter(brightness=0.2)]
    train_ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if strong_aug:
        train_ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), value='random'))

    eval_tf = transforms.Compose([
        transforms.Resize(size=resize_size),
        transforms.CenterCrop(size=img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    return {'train': transforms.Compose(train_ops), 'valid': eval_tf, 'test': eval_tf}


def build_display_transform(img_size=224):
    """与测试预处理几何一致但不归一化的变换 (用于 Grad-CAM 叠加显示, 保证热力图与图像对齐)"""
    resize_size = int(round(img_size * 256 / 224))
    return transforms.Compose([transforms.Resize(size=resize_size), transforms.CenterCrop(size=img_size)])


# 向后兼容: 其他脚本仍可直接 import image_transforms
image_transforms = build_transforms(224)


def split_indices(targets, seed=42):
    """固定随机种子分层划分: 70% Train, 15% Valid, 15% Test"""
    targets = np.asarray(targets)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, temp_idx = next(sss1.split(np.zeros(len(targets)), targets))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    valid_idx, test_idx = next(sss2.split(np.zeros(len(temp_idx)), targets[temp_idx]))
    return train_idx, temp_idx[valid_idx], temp_idx[test_idx]


def prepare_dataloaders(data_dir, batch_size=32, img_size=224, num_workers=4, strong_aug=True, split_seed=42):
    base_dataset = datasets.ImageFolder(root=data_dir)
    targets = np.array(base_dataset.targets)

    # 平方根平滑类别权重 (兼顾 trash 少数类召回率与整体精确率)
    class_counts = np.bincount(targets)
    raw_weights = 1.0 / torch.sqrt(torch.tensor(class_counts, dtype=torch.float))
    class_weights = raw_weights / raw_weights.sum() * len(class_counts)

    train_idx, valid_idx, test_idx = split_indices(targets, seed=split_seed)
    tfs = build_transforms(img_size, strong_aug=strong_aug)

    train_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=tfs['train']), train_idx)
    valid_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=tfs['valid']), valid_idx)
    test_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=tfs['test']), test_idx)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                         persistent_workers=num_workers > 0)
    # drop_last: 避免最后一个 batch 只有 1 张图导致 BN 报错 / Mixup 失效
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, valid_loader, test_loader, class_weights, base_dataset.classes, test_idx

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

# 结合两篇论文的数据增强规范
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def prepare_dataloaders(data_dir, batch_size=32):
    base_dataset = datasets.ImageFolder(root=data_dir)
    targets = np.array(base_dataset.targets)

    # 平方根平滑类别权重 (兼顾 trash 少数类召回率与整体精确率)
    class_counts = np.bincount(targets)
    raw_weights = 1.0 / torch.sqrt(torch.tensor(class_counts, dtype=torch.float))
    class_weights = raw_weights / raw_weights.sum() * len(class_counts)

    # 固定随机种子 42: 70% Train, 15% Valid, 15% Test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(sss1.split(np.zeros(len(targets)), targets))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    valid_idx, test_idx = next(sss2.split(np.zeros(len(temp_idx)), targets[temp_idx]))
    valid_idx, test_idx = temp_idx[valid_idx], temp_idx[test_idx]

    train_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=image_transforms['train']), train_idx)
    valid_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=image_transforms['valid']), valid_idx)
    test_dataset  = Subset(datasets.ImageFolder(root=data_dir, transform=image_transforms['test']), test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader, class_weights, base_dataset.classes, test_idx
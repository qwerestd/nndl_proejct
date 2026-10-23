import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
# ==========================================
# 1. 数据增强与预处理 (结合两篇论文的参数规范)
# ==========================================
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),  # ResSeMo 表2参数: ±15°
        transforms.ColorJitter(brightness=0.2),  # ResSeMo 表2参数: 亮度调整 ±20%
        transforms.RandomHorizontalFlip(p=0.5),  # RWC-Net: p=0.5 翻转
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


# ==========================================
# 2. 核心网络模块设计 (融合 ResSeMo 与 RWC-Net)
# ==========================================
class SEBlock(nn.Module):
    """ResSeMo 论文中的 SENet 通道注意力模块 (公式 2-4)"""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class HybridTrashNet(nn.Module):
    """
    结合 ResSeMo (ResNeXt + SENet + MobileNetV3)
    与 RWC-Net (双流融合 + 辅助分类头深层监督)
    """

    def __init__(self, num_classes=6, dropout_rate=0.2):
        super(HybridTrashNet, self).__init__()

        # 分支 1: ResNeXt50_32x4d (对应 ResSeMo 的分组卷积特征提取: 32 groups)
        resnext = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        self.branch1_features = nn.Sequential(*list(resnext.children())[:-2])  # 输出通道: 2048

        # 分支 2: MobileNetV3-Large (对应 ResSeMo 的轻量化倒残差与 h-swish)
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.branch2_features = mobilenet.features  # 输出通道: 960

        # 通道注意力模块 (SENet, r=16)
        self.se_block1 = SEBlock(2048, reduction=16)
        self.se_block2 = SEBlock(960, reduction=16)

        # 1x1 卷积特征融合 (ResSeMo 表2: Feature fusion convolution kernel 1x1)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2048 + 960, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Hardswish(inplace=True)
        )
        self.se_fusion = SEBlock(512, reduction=16)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # RWC-Net 辅助分类头 1 (接 ResNeXt 分支)
        self.aux_head1 = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

        # RWC-Net 辅助分类头 2 (接 MobileNetV3 分支)
        self.aux_head2 = nn.Sequential(
            nn.Linear(960, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

        # 主分类头 (融合特征输出)
        self.main_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # 1. 双分支特征提取 + SENet 增强
        f1 = self.se_block1(self.branch1_features(x))  # [B, 2048, 7, 7]
        f2 = self.se_block2(self.branch2_features(x))  # [B, 960, 7, 7]

        # 2. 跨模块特征拼接与 1x1 卷积压缩
        fused = torch.cat([f1, f2], dim=1)  # [B, 3008, 7, 7]
        fused = self.se_fusion(self.fusion_conv(fused))  # [B, 512, 7, 7]

        # 3. 池化展平
        v_main = torch.flatten(self.global_pool(fused), 1)
        out_main = self.main_classifier(v_main)

        # 训练模式下返回主输出与两个辅助分支输出，用于计算深层监督损失
        if self.training:
            v1 = torch.flatten(self.global_pool(f1), 1)
            v2 = torch.flatten(self.global_pool(f2), 1)
            out_aux1 = self.aux_head1(v1)
            out_aux2 = self.aux_head2(v2)
            return out_main, out_aux1, out_aux2

        # 推理/验证模式下仅返回主分类器结果
        return out_main


# ==========================================
# 3. 数据加载函数 (自动分层划分 Train/Valid/Test)
# ==========================================
def prepare_dataloaders(data_dir, batch_size=32):
    base_dataset = datasets.ImageFolder(root=data_dir)
    targets = np.array(base_dataset.targets)

    # 計算类别权重，解决 Trash/Litter 类别样本较少的问题 (ResSeMo 表2)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / torch.sqrt(torch.tensor(class_counts, dtype=torch.float))
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    # 按照 70% 训练, 15% 验证, 15% 测试进行分层随机划分
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(sss1.split(np.zeros(len(targets)), targets))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    valid_idx, test_idx = next(sss2.split(np.zeros(len(temp_idx)), targets[temp_idx]))
    valid_idx, test_idx = temp_idx[valid_idx], temp_idx[test_idx]

    train_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=image_transforms['train']), train_idx)
    valid_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=image_transforms['valid']), valid_idx)
    test_dataset = Subset(datasets.ImageFolder(root=data_dir, transform=image_transforms['test']), test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader, class_weights, base_dataset.classes


# ==========================================
# 4. 训练引擎 (含两阶段优化器切换与早停机制)
# ==========================================
def train_model(data_dir, save_path='./trashnet_best.pt', epochs=30, batch_size=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader, class_weights, class_names = prepare_dataloaders(data_dir, batch_size)
    model = HybridTrashNet(num_classes=len(class_names), dropout_rate=0.3).to(device)

    criterion = nn.NLLLoss(weight=class_weights.to(device))

    # 阶段 1: 差分学习率 Adam (保护预训练主干，防止第 6 轮后震荡)
    backbone_params = list(model.branch1_features.parameters()) + list(model.branch2_features.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith(('branch1_features', 'branch2_features'))]

    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': 1e-4},  # 预训练层用小学习率微调 (借鉴 RWC-Net)
        {'params': head_params, 'lr': 1e-3}  # 新增融合层与分类头快速收敛 (借鉴 ResSeMo)
    ], weight_decay=5e-4)

    # 第一阶段配合学习率衰减
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    best_valid_acc = 0.0
    patience_counter = 0
    patience_limit = 10
    switch_epoch = 12  # 在第 12 轮切换到 SGD 精细打磨

    for epoch in range(epochs):
        epoch_start = time.time()

        # 阶段 2: 切换为 SGD + 余弦退火，并回滚加载阶段 1 的最优模型权重
        if epoch == switch_epoch:
            print("\n>>> Loading best stage-1 weights & switching to SGD + Cosine Annealing...")
            model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
            optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - switch_epoch, eta_min=1e-6)
            patience_counter = 0  # 重置早停计数器，给 SGD 充足的微调空间

        # --- 训练阶段 ---
        model.train()
        train_loss, train_correct = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            out_main, out_aux1, out_aux2 = model(inputs)
            loss = criterion(out_main, labels) + 0.5 * criterion(out_aux1, labels) + 0.25 * criterion(out_aux2, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(out_main, 1)
            train_correct += torch.sum(preds == labels.data).item()

        scheduler.step()

        # --- 验证阶段 ---
        model.eval()
        valid_loss, valid_correct = 0.0, 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                valid_correct += torch.sum(preds == labels.data).item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_correct / len(train_loader.dataset)
        avg_valid_loss = valid_loss / len(valid_loader.dataset)
        avg_valid_acc = valid_correct / len(valid_loader.dataset)

        print(f"Epoch [{epoch + 1:02d}/{epochs:02d}] ({time.time() - epoch_start:.1f}s) | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc * 100:.2f}% | "
              f"Valid Loss: {avg_valid_loss:.4f}, Acc: {avg_valid_acc * 100:.2f}%")

        if avg_valid_acc > best_valid_acc:
            best_valid_acc = avg_valid_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Best model saved to {save_path} (Valid Acc: {best_valid_acc * 100:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    # 训练结束后，自动在独立测试集上生成论文级评估报告与混淆矩阵
    evaluate_on_testset(model, test_loader, save_path, class_names, device)


# ==========================================
# 5. 单张图片预测函数
# ==========================================
def make_prediction(model_path, image_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    model = HybridTrashNet(num_classes=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    test_image = Image.open(image_path).convert('RGB')
    tensor_image = image_transforms['test'](test_image).unsqueeze(0).to(device)

    with torch.no_grad():
        log_probs = model(tensor_image)
        probs = torch.exp(log_probs)
        top_prob, top_class = probs.topk(1, dim=1)
        pred_idx = top_class.item()
        confidence = top_prob.item() * 100

    plt.imshow(test_image)
    plt.title(f"Predicted: {class_names[pred_idx]} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

    return class_names[pred_idx], confidence

def evaluate_on_testset(model, test_loader, model_path, class_names, device):
    print("\n" + "="*50)
    print("Evaluating Best Model on Independent Test Set...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 打印每个类别的 Precision, Recall, F1-score (对应 RWC-Net 表 3)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # 绘制并保存混淆矩阵 (对应 ResSeMo 图 9 & RWC-Net 图 3)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix on TrashNet Test Set')
    plt.tight_layout()
    plt.savefig('./confusion_matrix.png', dpi=300)
    plt.show()
    print("Confusion matrix saved to ./confusion_matrix.png")

if __name__ == '__main__':
    # 自动定位你之前用 resize.py 生成的 dataset-resized 文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, 'data', 'dataset-resized')

    # 如果脚本本身就在 data 目录下，则直接找同级的 dataset-resized
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(current_dir, 'dataset-resized')

    model_save_path = os.path.join(current_dir, 'trashnet_best.pt')

    # 1. 启动训练
    if os.path.exists(dataset_path):
        train_model(data_dir=dataset_path, save_path=model_save_path, epochs=30, batch_size=32)
    else:
        print(f"请检查数据集路径是否存在: {dataset_path}")

    # 2. 预测示例 (训练完成后取消下方注释即可测试单张图片)
    pred_label, conf = make_prediction(model_save_path, './test/metal2.jpg')
    print(f"Prediction: {pred_label} ({conf:.2f}%)")


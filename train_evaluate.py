import os
import time
import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import prepare_dataloaders
from models import *

def train_single_model(model, model_name, train_loader, valid_loader, class_weights, save_path, epochs=30, device='cuda'):
    print(f"\n{'='*60}\nStart Training [{model_name}] on {device}\n{'='*60}")
    model = model.to(device)
    criterion = nn.NLLLoss(weight=class_weights.to(device))

    # 区分预训练主干与新增分类头，采用差分学习率
    backbone_keys = ('branch1_features', 'branch2_features', 'dense_stage', 'mobile_stage')
    backbone_params = [p for n, p in model.named_parameters() if n.startswith(backbone_keys)]
    head_params = [p for n, p in model.named_parameters() if not n.startswith(backbone_keys)]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': head_params, 'lr': 1e-3}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_acc, patience, patience_limit = 0.0, 0, 10

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss, train_correct = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # 主损失 + 指数衰减辅助分支损失 (论文 2 公式 2)
            out_main, out_aux2, out_aux1 = model(inputs)
            loss = criterion(out_main, labels) + 0.5 * criterion(out_aux2, labels) + 0.25 * criterion(out_aux1, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (out_main.argmax(1) == labels).sum().item()

        scheduler.step()

        # 验证集评估
        model.eval()
        valid_loss, valid_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                valid_loss += criterion(outputs, labels).item() * inputs.size(0)
                valid_correct += (outputs.argmax(1) == labels).sum().item()

        t_acc = train_correct / len(train_loader.dataset) * 100
        v_acc = valid_correct / len(valid_loader.dataset) * 100
        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] ({time.time()-t0:.1f}s) | "
              f"Train Loss: {train_loss/len(train_loader.dataset):.4f}, Acc: {t_acc:.2f}% | "
              f"Valid Loss: {valid_loss/len(valid_loader.dataset):.4f}, Acc: {v_acc:.2f}%")

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Best [{model_name}] saved (Valid Acc: {best_acc:.2f}%)")
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break


def get_tta_probabilities(model, test_loader, device):
    """使用测试时增强 (TTA: 原图 + 水平翻转) 获取预测概率分布"""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            p1 = torch.exp(model(inputs))
            p2 = torch.exp(model(torch.flip(inputs, dims=[3])))
            all_probs.append(((p1 + p2) / 2.0).cpu())
            all_labels.extend(labels.numpy())
    return torch.cat(all_probs, dim=0), all_labels


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


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'dataset-resized')
    out_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, test_loader, class_weights, class_names, _ = prepare_dataloaders(data_dir, batch_size=32)

    hybrid_ckpt = os.path.join(out_dir, 'hybrid_ressemo_best.pt')
    rwc_ckpt = os.path.join(out_dir, 'rwcnet_best.pt')

    # 1. 训练或加载模型 1 (HybridTrashNet - 论文 1)
    model_hybrid = HybridTrashNet(num_classes=len(class_names)).to(device)
    if not os.path.exists(hybrid_ckpt):
        train_single_model(model_hybrid, "Hybrid-ResSeMo", train_loader, valid_loader, class_weights, hybrid_ckpt, epochs=30, device=device)
    model_hybrid.load_state_dict(torch.load(hybrid_ckpt, map_location=device, weights_only=True))

    # 2. 训练或加载模型 2 (RWCNet - 论文 2)
    model_rwc = RWCNet(num_classes=len(class_names)).to(device)
    if not os.path.exists(rwc_ckpt):
        train_single_model(model_rwc, "RWC-Net", train_loader, valid_loader, class_weights, rwc_ckpt, epochs=12, device=device)
    model_rwc.load_state_dict(torch.load(rwc_ckpt, map_location=device, weights_only=True))

    # 3. 独立测试集评估与双模型 Ensemble 对比
    probs_hybrid, labels = get_tta_probabilities(model_hybrid, test_loader, device)
    probs_rwc, _         = get_tta_probabilities(model_rwc, test_loader, device)
    probs_ensemble = 0.5*probs_hybrid + 0.5*probs_rwc
    preds_hybrid   = probs_hybrid.argmax(dim=1).numpy()
    preds_rwc      = probs_rwc.argmax(dim=1).numpy()
    preds_ensemble = probs_ensemble.argmax(dim=1).numpy()

    print("\n" + "="*60)
    print(f"1. [Hybrid-ResSeMo] Test Accuracy: {accuracy_score(labels, preds_hybrid)*100:.2f}%")
    print(classification_report(labels, preds_hybrid, target_names=class_names, digits=4))
    save_confusion_matrix(labels, preds_hybrid, class_names, "Hybrid-ResSeMo Confusion Matrix", os.path.join(out_dir, 'cm_hybrid.png'), 'Blues')

    print("="*60)
    print(f"2. [RWC-Net] Test Accuracy: {accuracy_score(labels, preds_rwc)*100:.2f}%")
    print(classification_report(labels, preds_rwc, target_names=class_names, digits=4))
    save_confusion_matrix(labels, preds_rwc, class_names, "RWC-Net Confusion Matrix", os.path.join(out_dir, 'cm_rwcnet.png'), 'Greens')

    print("="*60)
    print(f"3. [Ensemble: RWC-Net + ResSeMo] Test Accuracy: {accuracy_score(labels, preds_ensemble)*100:.2f}%")
    print(classification_report(labels, preds_ensemble, target_names=class_names, digits=4))
    save_confusion_matrix(labels, preds_ensemble, class_names, "Ensemble Confusion Matrix", os.path.join(out_dir, 'cm_ensemble.png'), 'Purples')
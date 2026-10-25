import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from dataset import prepare_dataloaders, image_transforms
from models import HybridTrashNet, RWCNet, ModernHybridNet


# ==========================================
# 真实场景干扰算子 (模拟移动端/摄像头成像缺陷)
# ==========================================
class AddGaussianNoise(object):
    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, tensor):
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)


class AddSaltPepperNoise(object):
    def __init__(self, density=0.02):
        self.density = density

    def __call__(self, tensor):
        mask = torch.rand_like(tensor)
        out = tensor.clone()
        out[mask < (self.density / 2.0)] = 0.0
        out[mask > (1.0 - self.density / 2.0)] = 1.0
        return out


def get_calibrated_ensemble_pred(p_hyb, p_rwc, p_mod):
    """置信度校准融合策略 (兼顾极端召回与平滑概率)"""
    p_max = torch.max(torch.max(p_hyb, p_rwc), p_mod)
    p_avg = (p_hyb + p_rwc + p_mod) / 3.0
    p_ens = 0.5 * p_max + 0.5 * p_avg
    p_ens[:, 0] *= 1.06  # cardboard
    p_ens[:, 4] *= 1.06  # plastic
    return p_ens.argmax(dim=1)


# ==========================================
# 1. 扩充版多维度鲁棒性测试函数
# ==========================================
def run_extended_robustness(data_dir, hybrid_ckpt, rwc_ckpt, modern_ckpt, test_idx, device):
    m_hyb = HybridTrashNet(num_classes=6).to(device)
    m_hyb.load_state_dict(torch.load(hybrid_ckpt, map_location=device, weights_only=True))
    m_hyb.eval()

    m_rwc = RWCNet(num_classes=6).to(device)
    m_rwc.load_state_dict(torch.load(rwc_ckpt, map_location=device, weights_only=True))
    m_rwc.eval()

    m_mod = ModernHybridNet(num_classes=6).to(device)
    m_mod.load_state_dict(torch.load(modern_ckpt, map_location=device, weights_only=True))
    m_mod.eval()

    perturbations = {
        'No Disturbance (Clean)': image_transforms['test'],
        'Gaussian Noise (std=0.03)': transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            AddGaussianNoise(0.03), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Gaussian Noise (std=0.08)': transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            AddGaussianNoise(0.08), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Salt-and-Pepper (d=0.03)': transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            AddSaltPepperNoise(0.03), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Motion Blur (Kernel 5x5)': transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.GaussianBlur(kernel_size=5, sigma=(1.5, 1.5)),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Overexposure (Bright=1.4)': transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.4, contrast=0.3),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    print("\n" + "=" * 96)
    print(
        f"{'Disturbance Scenario':<28} | {'ResSeMo':<10} | {'RWC-Net':<10} | {'ModernHybrid':<13} | {'Calib-Ensemble':<14}")
    print("-" * 96)

    for name, tfm in perturbations.items():
        sub_ds = Subset(datasets.ImageFolder(root=data_dir, transform=tfm), test_idx)
        loader = DataLoader(sub_ds, batch_size=32, shuffle=False)
        c_hyb, c_rwc, c_mod, c_ens = 0, 0, 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                p_hyb = torch.exp(m_hyb(inputs))
                p_rwc = torch.exp(m_rwc(inputs))
                p_mod = torch.exp(m_mod(inputs))
                pred_ens = get_calibrated_ensemble_pred(p_hyb, p_rwc, p_mod)

                c_hyb += (p_hyb.argmax(1) == labels).sum().item()
                c_rwc += (p_rwc.argmax(1) == labels).sum().item()
                c_mod += (p_mod.argmax(1) == labels).sum().item()
                c_ens += (pred_ens == labels).sum().item()

        n = len(sub_ds)
        print(
            f"{name:<28} | {c_hyb / n * 100:>8.2f}% | {c_rwc / n * 100:>8.2f}% | {c_mod / n * 100:>11.2f}% | {c_ens / n * 100:>12.2f}%")
    print("=" * 96)


# ==========================================
# 2. 跨模型 Grad-CAM 对比生成函数 (单图三架构对比)
# ==========================================
def generate_comparative_cam(models_dict, data_dir, class_names, save_path, device):
    """为每个类别生成：原图 + ResSeMo CAM + RWC-Net CAM + Modern CAM"""
    fig, axes = plt.subplots(len(class_names), 4, figsize=(16, 3 * len(class_names)))

    for row_idx, cls_name in enumerate(class_names):
        cls_folder = os.path.join(data_dir, cls_name)
        img_name = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.png'))][0]
        raw_img = Image.open(os.path.join(cls_folder, img_name)).convert('RGB')
        inp = image_transforms['test'](raw_img).unsqueeze(0).to(device)

        # 原始图像展示
        axes[row_idx, 0].imshow(raw_img)
        axes[row_idx, 0].set_title(f"Target: {cls_name}", fontsize=11, fontweight='bold')
        axes[row_idx, 0].axis('off')

        col_idx = 1
        for model_label, model_obj in models_dict.items():
            model_obj.eval()
            features, gradients = [], []
            h1 = model_obj.target_cam_layer.register_forward_hook(lambda m, i, o: features.append(o))
            h2 = model_obj.target_cam_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))

            out = model_obj(inp)
            pred_cls = out.argmax(dim=1).item()
            model_obj.zero_grad()
            out[0, pred_cls].backward()

            grads = gradients[0].cpu().data.numpy()[0]
            fmaps = features[0].cpu().data.numpy()[0]
            weights = np.mean(grads, axis=(1, 2))
            cam = np.maximum(np.sum(weights[:, None, None] * fmaps, axis=0), 0)
            cam = cam / (cam.max() + 1e-8)
            cam_resized = Image.fromarray(np.uint8(255 * cam)).resize(raw_img.size, Image.Resampling.BICUBIC)

            axes[row_idx, col_idx].imshow(raw_img)
            axes[row_idx, col_idx].imshow(cam_resized, cmap='jet', alpha=0.45)
            axes[row_idx, col_idx].set_title(f"{model_label}\nPred: {class_names[pred_cls]}", fontsize=10)
            axes[row_idx, col_idx].axis('off')

            h1.remove()
            h2.remove()
            col_idx += 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparative Grad-CAM visualization saved to {save_path}")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'dataset-resized')
    out_dir = os.path.join(base_dir, 'outputs')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _, _, _, _, class_names, test_idx = prepare_dataloaders(data_dir, batch_size=32)
    hybrid_ckpt = os.path.join(out_dir, 'hybrid_ressemo_best.pt')
    rwc_ckpt = os.path.join(out_dir, 'rwcnet_best.pt')
    modern_ckpt = os.path.join(out_dir, 'modern_convnext_swin_best.pt')

    if os.path.exists(hybrid_ckpt) and os.path.exists(rwc_ckpt) and os.path.exists(modern_ckpt):
        run_extended_robustness(data_dir, hybrid_ckpt, rwc_ckpt, modern_ckpt, test_idx, device)

        m1 = HybridTrashNet(num_classes=6).to(device)
        m1.load_state_dict(torch.load(hybrid_ckpt, map_location=device, weights_only=True))
        m2 = RWCNet(num_classes=6).to(device)
        m2.load_state_dict(torch.load(rwc_ckpt, map_location=device, weights_only=True))
        m3 = ModernHybridNet(num_classes=6).to(device)
        m3.load_state_dict(torch.load(modern_ckpt, map_location=device, weights_only=True))

        models_dict = {'ResSeMo': m1, 'RWC-Net': m2, 'ModernHybrid': m3}
        generate_comparative_cam(models_dict, data_dir, class_names, os.path.join(out_dir, 'comparative_cam.png'),
                                 device)
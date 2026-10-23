import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from dataset import prepare_dataloaders, image_transforms
from models import HybridTrashNet, RWCNet

class AddGaussianNoise(object):
    def __init__(self, std=0.05): self.std = std
    def __call__(self, tensor): return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)

class AddSaltPepperNoise(object):
    def __init__(self, density=0.02): self.density = density
    def __call__(self, tensor):
        mask = torch.rand_like(tensor)
        out = tensor.clone()
        out[mask < (self.density / 2)] = 0.0
        out[mask > 1 - (self.density / 2)] = 1.0
        return out

def run_robustness_comparison(data_dir, hybrid_ckpt, rwc_ckpt, test_idx, device):
    """对比两个模型在 4 种环境噪声扰动下的抗干扰能力 (对标论文 1 表 9)"""
    m_hybrid = HybridTrashNet(num_classes=6).to(device)
    m_hybrid.load_state_dict(torch.load(hybrid_ckpt, map_location=device, weights_only=True))
    m_hybrid.eval()

    m_rwc = RWCNet(num_classes=6).to(device)
    m_rwc.load_state_dict(torch.load(rwc_ckpt, map_location=device, weights_only=True))
    m_rwc.eval()

    perturbations = {
        'No Disturbance': image_transforms['test'],
        'Gaussian Noise (0.05)': transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            AddGaussianNoise(0.05), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Salt-and-Pepper (0.02)': transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            AddSaltPepperNoise(0.02), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'Random Blur (3x3)': transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    print("\n" + "="*75)
    print(f"{'Disturbance Scenario':<26} | {'Hybrid-ResSeMo':<15} | {'RWC-Net':<12} | {'Ensemble':<12}")
    print("-" * 75)

    for name, tfm in perturbations.items():
        sub_ds = Subset(datasets.ImageFolder(root=data_dir, transform=tfm), test_idx)
        loader = DataLoader(sub_ds, batch_size=32, shuffle=False)
        c_hyb, c_rwc, c_ens = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                p_hyb = torch.exp(m_hybrid(inputs))
                p_rwc = torch.exp(m_rwc(inputs))
                p_ens = 0.45 * p_hyb + 0.55 * p_rwc

                c_hyb += (p_hyb.argmax(1) == labels).sum().item()
                c_rwc += (p_rwc.argmax(1) == labels).sum().item()
                c_ens += (p_ens.argmax(1) == labels).sum().item()

        n = len(sub_ds)
        print(f"{name:<26} | {c_hyb/n*100:>13.2f}% | {c_rwc/n*100:>10.2f}% | {c_ens/n*100:>10.2f}%")
    print("="*75)


def generate_cam_grid(model, model_ckpt, data_dir, class_names, save_path, device):
    """为 6 个垃圾类别各抽取一张代表图像生成 CAM 热力图 (对标论文 2 图 4)"""
    model.load_state_dict(torch.load(model_ckpt, map_location=device, weights_only=True))
    model.eval()

    features, gradients = [], []
    h1 = model.target_cam_layer.register_forward_hook(lambda m, i, o: features.append(o))
    h2 = model.target_cam_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0]))

    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    for idx, cls_name in enumerate(class_names):
        cls_folder = os.path.join(data_dir, cls_name)
        img_file = os.path.join(cls_folder, os.listdir(cls_folder)[0])
        raw_img = Image.open(img_file).convert('RGB')
        inp = image_transforms['test'](raw_img).unsqueeze(0).to(device)

        features.clear()
        gradients.clear()
        out = model(inp)
        pred_cls = out.argmax(dim=1).item()
        model.zero_grad()
        out[0, pred_cls].backward()

        grads = gradients[0].cpu().data.numpy()[0]
        fmaps = features[0].cpu().data.numpy()[0]
        weights = np.mean(grads, axis=(1, 2))
        cam = np.maximum(np.sum(weights[:, None, None] * fmaps, axis=0), 0)
        cam = cam / (cam.max() + 1e-8)
        cam_resized = Image.fromarray(np.uint8(255 * cam)).resize(raw_img.size, Image.Resampling.BICUBIC)

        axes[0, idx].imshow(raw_img)
        axes[0, idx].set_title(f"True: {cls_name}")
        axes[0, idx].axis('off')

        axes[1, idx].imshow(raw_img)
        axes[1, idx].imshow(cam_resized, cmap='jet', alpha=0.45)
        axes[1, idx].set_title(f"CAM: {class_names[pred_cls]}")
        axes[1, idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    h1.remove()
    h2.remove()
    print(f"6-Class CAM grid saved to {save_path}")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'dataset-resized')
    out_dir = os.path.join(base_dir, 'outputs')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _, _, _, _, class_names, test_idx = prepare_dataloaders(data_dir, batch_size=32)
    hybrid_ckpt = os.path.join(out_dir, 'hybrid_ressemo_best.pt')
    rwc_ckpt = os.path.join(out_dir, 'rwcnet_best.pt')

    if os.path.exists(hybrid_ckpt) and os.path.exists(rwc_ckpt):
        run_robustness_comparison(data_dir, hybrid_ckpt, rwc_ckpt, test_idx, device)
        generate_cam_grid(RWCNet(num_classes=6).to(device), rwc_ckpt, data_dir, class_names, os.path.join(out_dir, 'cam_visualization.png'), device)
    else:
        print("请先运行 train_evaluate.py 生成两个模型的权重文件！")
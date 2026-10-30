import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from dataset import prepare_dataloaders, build_transforms, build_display_transform, IMAGENET_MEAN, IMAGENET_STD
from ensemble import ensemble_probs, load_ensemble_config
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


def get_calibrated_ensemble_probs(logits_list, ens_cfg):
    """使用验证集拟合的 温度 / 权重 / alpha / 类别偏置 进行融合 (与 train_evaluate.py 及 ONNX 导出完全一致)"""
    probs = [F.softmax(l / t, dim=1) for l, t in zip(logits_list, ens_cfg['temperatures'])]
    return ensemble_probs(probs, ens_cfg['weights'], ens_cfg['alpha'], ens_cfg['class_bias'])


def load_models(ckpts, device, num_classes=6):
    builders = [HybridTrashNet, RWCNet, ModernHybridNet]
    loaded = []
    for build, ckpt in zip(builders, ckpts):
        kwargs = {}
        if build is ModernHybridNet:
            # 根据 checkpoint 文件名推断融合方式 (train_evaluate.py --modern-fusion 的命名规则)
            base = os.path.basename(ckpt)
            kwargs['fusion'] = 'cross_attn' if 'cross_attn' in base else ('gated' if 'gated' in base else 'concat')
        m = build(num_classes=num_classes, **kwargs).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        loaded.append(m.eval())
    return loaded


# ==========================================
# 1. 扩充版多维度鲁棒性测试函数
# ==========================================
def run_extended_robustness(data_dir, model_list, ens_cfg, test_idx, device, img_size=224):
    m_hyb, m_rwc, m_mod = model_list
    rs = int(round(img_size * 256 / 224))
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    perturbations = {
        'No Disturbance (Clean)': build_transforms(img_size)['test'],
        'Gaussian Noise (std=0.03)': transforms.Compose([
            transforms.Resize(rs), transforms.CenterCrop(img_size), transforms.ToTensor(),
            AddGaussianNoise(0.03), norm
        ]),
        'Gaussian Noise (std=0.08)': transforms.Compose([
            transforms.Resize(rs), transforms.CenterCrop(img_size), transforms.ToTensor(),
            AddGaussianNoise(0.08), norm
        ]),
        'Salt-and-Pepper (d=0.03)': transforms.Compose([
            transforms.Resize(rs), transforms.CenterCrop(img_size), transforms.ToTensor(),
            AddSaltPepperNoise(0.03), norm
        ]),
        'Motion Blur (Kernel 5x5)': transforms.Compose([
            transforms.Resize(rs), transforms.CenterCrop(img_size),
            transforms.GaussianBlur(kernel_size=5, sigma=(1.5, 1.5)),
            transforms.ToTensor(), norm
        ]),
        'Overexposure (Bright=1.4)': transforms.Compose([
            transforms.Resize(rs), transforms.CenterCrop(img_size),
            transforms.ColorJitter(brightness=0.4, contrast=0.3),
            transforms.ToTensor(), norm
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
                l_hyb, l_rwc, l_mod = m_hyb(inputs), m_rwc(inputs), m_mod(inputs)
                pred_ens = get_calibrated_ensemble_probs([l_hyb, l_rwc, l_mod], ens_cfg).argmax(1)

                c_hyb += (l_hyb.argmax(1) == labels).sum().item()
                c_rwc += (l_rwc.argmax(1) == labels).sum().item()
                c_mod += (l_mod.argmax(1) == labels).sum().item()
                c_ens += (pred_ens == labels).sum().item()

        n = len(sub_ds)
        print(
            f"{name:<28} | {c_hyb / n * 100:>8.2f}% | {c_rwc / n * 100:>8.2f}% | {c_mod / n * 100:>11.2f}% | {c_ens / n * 100:>12.2f}%")
    print("=" * 96)


# ==========================================
# 2. Grad-CAM 样本选择 (只从测试集中选, 避免展示训练集图片)
# ==========================================
@torch.no_grad()
def select_cam_samples(data_dir, test_idx, model_list, ens_cfg, device, img_size=224, max_errors=8):
    """
    返回:
      correct_samples: 每类 1 张三个模型都预测正确的测试图 (若不存在则退回集成正确的样本)
      error_samples  : 至多 max_errors 张被校准集成分错的测试图 (按置信度降序, 优先展示 "自信地犯错")
    每个元素为 (图片路径, 真实标签, 集成预测)
    """
    ds = datasets.ImageFolder(root=data_dir, transform=build_transforms(img_size)['test'])
    loader = DataLoader(Subset(ds, test_idx), batch_size=32, shuffle=False)
    logits = [[], [], []]
    labels_all = []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        for i, m in enumerate(model_list):
            logits[i].append(m(inputs).cpu())
        labels_all.append(labels)
    logits = [torch.cat(l) for l in logits]
    labels_all = torch.cat(labels_all)
    p_ens = get_calibrated_ensemble_probs(logits, ens_cfg)
    conf_ens, pred_ens = p_ens.max(1)
    all_correct = torch.stack([l.argmax(1) == labels_all for l in logits]).all(0)
    paths = [ds.samples[i][0] for i in test_idx]

    correct_samples = []
    for c in range(len(ds.classes)):
        idx = torch.where((labels_all == c) & all_correct)[0]
        if len(idx) == 0:
            idx = torch.where((labels_all == c) & (pred_ens == labels_all))[0]
        if len(idx) == 0:
            idx = torch.where(labels_all == c)[0]
        i = idx[0].item()
        correct_samples.append((paths[i], c, pred_ens[i].item()))

    err_idx = torch.where(pred_ens != labels_all)[0]
    err_idx = err_idx[conf_ens[err_idx].argsort(descending=True)][:max_errors]
    error_samples = [(paths[i], labels_all[i].item(), pred_ens[i].item()) for i in err_idx.tolist()]
    return correct_samples, error_samples


def _grad_cam(model, inp, target_class=None):
    """基于 target_cam_layer 的 Grad-CAM; 使用张量 hook, 对 SE / Identity 层均适用"""
    feats, grads = [], []

    def fwd_hook(_m, _i, o):
        feats.append(o)
        o.register_hook(lambda g: grads.append(g))

    h = model.target_cam_layer.register_forward_hook(fwd_hook)
    try:
        model.eval()
        model.zero_grad()
        logits = model(inp)
        pred_cls = logits.argmax(dim=1).item()
        cls = pred_cls if target_class is None else target_class
        logits[0, cls].backward()  # 对 logit 求梯度 (比 log-prob 更符合 Grad-CAM 原始定义)
    finally:
        h.remove()

    fmaps = feats[0].detach()[0]
    weights = grads[0].detach()[0].mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * fmaps).sum(0))
    cam = (cam / (cam.max() + 1e-8)).cpu().numpy()
    return cam, pred_cls


def generate_comparative_cam(models_dict, samples, class_names, save_path, device, img_size=224, title_prefix=''):
    """每行一张测试图: 原图(中心裁剪, 与模型输入对齐) + 各模型的 Grad-CAM"""
    test_tf = build_transforms(img_size)['test']
    disp_tf = build_display_transform(img_size)
    n_cols = len(models_dict) + 1
    fig, axes = plt.subplots(len(samples), n_cols, figsize=(4 * n_cols, 3.2 * len(samples)), squeeze=False)

    for row_idx, (path, true_cls, ens_pred) in enumerate(samples):
        raw_img = Image.open(path).convert('RGB')
        disp_img = disp_tf(raw_img)
        inp = test_tf(raw_img).unsqueeze(0).to(device)

        axes[row_idx, 0].imshow(disp_img)
        axes[row_idx, 0].set_title(f"{title_prefix}True: {class_names[true_cls]}\nEnsemble: {class_names[ens_pred]}",
                                   fontsize=11, fontweight='bold',
                                   color='black' if true_cls == ens_pred else 'red')
        axes[row_idx, 0].axis('off')

        for col_idx, (model_label, model_obj) in enumerate(models_dict.items(), start=1):
            cam, pred_cls = _grad_cam(model_obj, inp)
            cam_resized = Image.fromarray(np.uint8(255 * cam)).resize(disp_img.size, Image.Resampling.BICUBIC)
            axes[row_idx, col_idx].imshow(disp_img)
            axes[row_idx, col_idx].imshow(cam_resized, cmap='jet', alpha=0.45)
            axes[row_idx, col_idx].set_title(f"{model_label}\nPred: {class_names[pred_cls]}", fontsize=10,
                                             color='black' if pred_cls == true_cls else 'red')
            axes[row_idx, col_idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Comparative Grad-CAM visualization saved to {save_path}")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'dataset-resized')
    out_dir = os.path.join(base_dir, 'outputs')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ens_cfg = load_ensemble_config(os.path.join(out_dir, 'ensemble_config.json'))
    img_size = ens_cfg.get('img_size', 224)
    _, _, _, _, class_names, test_idx = prepare_dataloaders(data_dir, batch_size=32, img_size=img_size,
                                                            num_workers=0)
    ckpt_names = ens_cfg.get('checkpoints',
                             ['hybrid_ressemo_best.pt', 'rwcnet_best.pt', 'modern_convnext_swin_best.pt'])
    ckpts = [os.path.join(out_dir, c) for c in ckpt_names]

    if all(os.path.exists(c) for c in ckpts):
        model_list = load_models(ckpts, device, num_classes=len(class_names))
        run_extended_robustness(data_dir, model_list, ens_cfg, test_idx, device, img_size)

        models_dict = dict(zip(['ResSeMo', 'RWC-Net', 'ModernHybrid'], model_list))
        correct_samples, error_samples = select_cam_samples(data_dir, test_idx, model_list, ens_cfg, device,
                                                            img_size)
        generate_comparative_cam(models_dict, correct_samples, class_names,
                                 os.path.join(out_dir, 'comparative_cam.png'), device, img_size)
        if error_samples:
            generate_comparative_cam(models_dict, error_samples, class_names,
                                     os.path.join(out_dir, 'comparative_cam_errors.png'), device, img_size,
                                     title_prefix='[ERROR] ')
        else:
            print("集成模型在测试集上没有分错的样本, 跳过错误样本 Grad-CAM。")
    else:
        print(f"缺少 checkpoint: {[c for c in ckpts if not os.path.exists(c)]}, 请先运行 train_evaluate.py")

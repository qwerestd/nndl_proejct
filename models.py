import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ensemble import ensemble_probs


# 说明: 所有模型现在输出 logits (不再内置 LogSoftmax), 训练使用 CrossEntropyLoss(label_smoothing),
#       推理用 softmax 得到概率。旧 checkpoint 仍可直接加载 (LogSoftmax 无参数且位于 Sequential 末尾)。

def _w(weights, pretrained):
    return weights if pretrained else None


def _rescale_stochastic_depth(module, default_prob, target_prob):
    """按比例缩放已构建网络中所有 StochasticDepth 的丢弃概率 (保持逐层线性递增的分布)"""
    from torchvision.ops import StochasticDepth
    scale = target_prob / default_prob if default_prob > 0 else 0.0
    for m in module.modules():
        if isinstance(m, StochasticDepth):
            m.p = m.p * scale


# ==========================================
# 基础组件 1: SENet 通道注意力模块 (论文 1 核心组件)
# ==========================================
class SEBlock(nn.Module):
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


# ==========================================
# 基础组件 2 [Novel Idea]: ConvNeXt V2 全局响应归一化 (GRN)
# 作用: 解决双流特征拼接时的通道冗余，增强材质纹理特征竞争
# ==========================================
class GRN(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(GRN, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


# ==========================================
# 基础组件 3 [Fusion]: 门控融合 (Gated Fusion)
# 作用: 逐位置、逐通道地学习 CNN 纹理特征与 Transformer 全局特征的权重, 替代简单拼接
# ==========================================
class GatedFusion(nn.Module):
    def __init__(self, c1, c2, out_channels=512, use_grn=True):
        super(GatedFusion, self).__init__()
        self.proj1 = nn.Sequential(nn.Conv2d(c1, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.proj2 = nn.Sequential(nn.Conv2d(c2, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.gate = nn.Sequential(nn.Conv2d(out_channels * 2, out_channels, 1), nn.Sigmoid())
        self.post = nn.Sequential(nn.GELU(), GRN(out_channels) if use_grn else nn.Identity())

    def forward(self, f1, f2):
        p1, p2 = self.proj1(f1), self.proj2(f2)
        g = self.gate(torch.cat([p1, p2], dim=1))
        return self.post(g * p1 + (1.0 - g) * p2)


# ==========================================
# 基础组件 4 [Fusion]: 双向交叉注意力融合 (Bidirectional Cross-Attention)
# 作用: CNN token 查询 Transformer token (反之亦然), 显式建模两路特征之间的空间对应关系
# ==========================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, c1, c2, out_channels=512, num_heads=8, dropout=0.1, use_grn=True):
        super(CrossAttentionFusion, self).__init__()
        d = out_channels
        self.proj1 = nn.Conv2d(c1, d, 1)
        self.proj2 = nn.Conv2d(c2, d, 1)
        self.norm_q1, self.norm_kv1 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.norm_q2, self.norm_kv2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn_1to2 = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.attn_2to1 = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.Conv2d(2 * d, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            GRN(out_channels) if use_grn else nn.Identity()
        )

    def forward(self, f1, f2):
        p1, p2 = self.proj1(f1), self.proj2(f2)
        b, d, h, w = p1.shape
        t1 = p1.flatten(2).transpose(1, 2)  # (B, HW, D)
        t2 = p2.flatten(2).transpose(1, 2)
        kv2, kv1 = self.norm_kv2(t2), self.norm_kv1(t1)
        t1 = t1 + self.attn_1to2(self.norm_q1(t1), kv2, kv2, need_weights=False)[0]
        t2 = t2 + self.attn_2to1(self.norm_q2(t2), kv1, kv1, need_weights=False)[0]
        t1 = t1.transpose(1, 2).reshape(b, d, h, w)
        t2 = t2.transpose(1, 2).reshape(b, d, h, w)
        return self.out(torch.cat([t1, t2], dim=1))


# ==========================================
# 模型 1 [SOTA Baseline 1]: HybridTrashNet (ResNeXt50 + MobileNetV3)
# ==========================================
class HybridTrashNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.3, pretrained=True):
        super(HybridTrashNet, self).__init__()
        resnext = models.resnext50_32x4d(weights=_w(models.ResNeXt50_32X4D_Weights.DEFAULT, pretrained))
        self.branch1_features = nn.Sequential(*list(resnext.children())[:-2])  # 2048

        mobilenet = models.mobilenet_v3_large(weights=_w(models.MobileNet_V3_Large_Weights.DEFAULT, pretrained))
        self.branch2_features = mobilenet.features  # 960

        self.se_block1 = SEBlock(2048, reduction=16)
        self.se_block2 = SEBlock(960, reduction=16)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2048 + 960, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Hardswish(inplace=True)
        )
        self.target_cam_layer = SEBlock(512, reduction=16)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.aux_head1 = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        self.aux_head2 = nn.Sequential(
            nn.Linear(960, 256), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        self.main_classifier = nn.Sequential(
            nn.Linear(512, 256), nn.Hardswish(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f1 = self.se_block1(self.branch1_features(x))
        f2 = self.se_block2(self.branch2_features(x))

        fused = torch.cat([f1, f2], dim=1)
        fused = self.target_cam_layer(self.fusion_conv(fused))

        v_main = torch.flatten(self.global_pool(fused), 1)
        out_main = self.main_classifier(v_main)

        if self.training:
            v1 = torch.flatten(self.global_pool(f1), 1)
            v2 = torch.flatten(self.global_pool(f2), 1)
            # 返回顺序: (主输出, ResNeXt 分支辅助输出, MobileNet 分支辅助输出)
            return out_main, self.aux_head1(v1), self.aux_head2(v2)
        return out_main


# ==========================================
# 模型 2 [SOTA Baseline 2]: RWCNet (DenseNet201 + MobileNetV2 跨层深层监督)
# ==========================================
class RWCAuxiliaryBranch(nn.Module):
    def __init__(self, in_channels, num_classes=6, dropout_rate=0.3):
        super(RWCAuxiliaryBranch, self).__init__()
        self.se = SEBlock(in_channels, reduction=16)
        self.features = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3, padding=2),
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(self.se(x)))


class RWCNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.3, pretrained=True):
        super(RWCNet, self).__init__()
        df = models.densenet201(weights=_w(models.DenseNet201_Weights.DEFAULT, pretrained)).features
        self.dense_stage1 = nn.Sequential(df.conv0, df.norm0, df.relu0, df.pool0, df.denseblock1, df.transition1,
                                          df.denseblock2)  # 512
        self.dense_stage2 = nn.Sequential(df.transition2, df.denseblock3)  # 1792
        self.dense_stage3 = nn.Sequential(df.transition3, df.denseblock4, df.norm5)  # 1920

        mf = models.mobilenet_v2(weights=_w(models.MobileNet_V2_Weights.DEFAULT, pretrained)).features
        self.mobile_stage1 = nn.Sequential(*mf[:7])  # 32
        self.mobile_stage2 = nn.Sequential(*mf[7:18])  # 320
        self.mobile_stage3 = mf[18]  # 1280

        self.aux_branch1 = RWCAuxiliaryBranch(512 + 32, num_classes, dropout_rate)
        self.aux_branch2 = RWCAuxiliaryBranch(1792 + 320, num_classes, dropout_rate)

        self.target_cam_layer = SEBlock(1920 + 1280, reduction=16)
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1920 + 1280, 512), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        d1 = self.dense_stage1(x)
        m1 = self.mobile_stage1(x)
        d2 = self.dense_stage2(d1)
        m2 = self.mobile_stage2(m1)
        d3 = F.relu(self.dense_stage3(d2), inplace=True)
        m3 = self.mobile_stage3(m2)

        f_final = self.target_cam_layer(torch.cat([d3, m3], dim=1))
        out_final = self.final_classifier(self.final_pool(f_final))

        if self.training:
            out_aux1 = self.aux_branch1(torch.cat([d1, m1], dim=1))
            m2_up = F.interpolate(m2, size=d2.shape[2:], mode='bilinear', align_corners=False)
            out_aux2 = self.aux_branch2(torch.cat([d2, m2_up], dim=1))
            # 返回顺序: (主输出, 深层辅助输出, 浅层辅助输出)
            return out_final, out_aux2, out_aux1
        return out_final


# ==========================================
# 模型 3 [Proposed Novel SOTA]: ModernHybridNet (ConvNeXt + Swin Transformer)
# 所有开关默认值 = 原始完整模型 (旧 checkpoint 可直接加载); 修改开关用于消融实验
#   branches : 'both' | 'convnext' | 'swin'       单分支消融
#   fusion   : 'concat' | 'gated' | 'cross_attn'  融合方式
#   use_se / use_grn / use_aux                    组件消融
# ==========================================
class ModernHybridNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.3, branches='both', fusion='concat',
                 use_se=True, use_grn=True, use_aux=True,
                 drop_path_convnext=0.1, drop_path_swin=0.2, pretrained=True):
        super(ModernHybridNet, self).__init__()
        assert branches in ('both', 'convnext', 'swin')
        assert fusion in ('concat', 'gated', 'cross_attn')
        self.branches, self.fusion_type = branches, fusion
        self.use_aux = use_aux and branches == 'both'
        dim = 768

        # 分支 1: 现代化大核纯卷积 ConvNeXt-Tiny (7x7 感受野解决 cardboard/paper 混淆)
        if branches in ('both', 'convnext'):
            convnext = models.convnext_tiny(weights=_w(models.ConvNeXt_Tiny_Weights.DEFAULT, pretrained),
                                            stochastic_depth_prob=drop_path_convnext)
            self.branch1_features = convnext.features  # 768
            self.se_block1 = SEBlock(dim, reduction=16) if use_se else nn.Identity()

        # 分支 2: 视觉 Transformer Swin-Tiny (全局注意力解决 plastic/glass 反光混淆)
        if branches in ('both', 'swin'):
            swin = models.swin_t(weights=_w(models.Swin_T_Weights.DEFAULT, pretrained))
            _rescale_stochastic_depth(swin, default_prob=0.2, target_prob=drop_path_swin)  # swin_t 内部写死 0.2
            self.branch2_features = swin.features
            self.swin_norm = swin.norm  # 768
            self.se_block2 = SEBlock(dim, reduction=16) if use_se else nn.Identity()

        if branches == 'both' and fusion == 'gated':
            self.fusion = GatedFusion(dim, dim, 512, use_grn=use_grn)
        elif branches == 'both' and fusion == 'cross_attn':
            self.fusion = CrossAttentionFusion(dim, dim, 512, use_grn=use_grn)
        else:
            # 'concat' (或单分支时的投影); 保持属性名 fusion_conv 以兼容旧 checkpoint
            in_ch = dim * 2 if branches == 'both' else dim
            layers = [nn.Conv2d(in_ch, 512, kernel_size=1, bias=False), nn.BatchNorm2d(512), nn.GELU()]
            if use_grn:
                layers.append(GRN(512))  # 引入 ConvNeXt V2 全局响应归一化
            self.fusion_conv = nn.Sequential(*layers)

        self.target_cam_layer = SEBlock(512, reduction=16) if use_se else nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        if self.use_aux:
            self.aux_head1 = nn.Sequential(
                nn.Linear(dim, 256), nn.GELU(), nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            )
            self.aux_head2 = nn.Sequential(
                nn.Linear(dim, 256), nn.GELU(), nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            )
        self.main_classifier = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f1 = f2 = None
        if self.branches in ('both', 'convnext'):
            f1 = self.se_block1(self.branch1_features(x))
        if self.branches in ('both', 'swin'):
            f2 = self.swin_norm(self.branch2_features(x)).permute(0, 3, 1, 2).contiguous()
            f2 = self.se_block2(f2)

        if self.branches == 'both':
            if self.fusion_type == 'concat':
                fused = self.fusion_conv(torch.cat([f1, f2], dim=1))
            else:
                fused = self.fusion(f1, f2)
        else:
            fused = self.fusion_conv(f1 if f1 is not None else f2)

        fused = self.target_cam_layer(fused)
        v_main = torch.flatten(self.global_pool(fused), 1)
        out_main = self.main_classifier(v_main)

        if self.training and self.use_aux:
            v1 = torch.flatten(self.global_pool(f1), 1)
            v2 = torch.flatten(self.global_pool(f2), 1)
            # 返回顺序: (主输出, ConvNeXt 分支辅助输出, Swin 分支辅助输出)
            return out_main, self.aux_head1(v1), self.aux_head2(v2)
        return out_main


# ==========================================
# 组件 5 [Deployment Merit 专用]: 移动端部署与校准集成封装器
# 作用: 将任意数量的模型 + 验证集拟合的 温度 / 权重 / alpha / 类别偏置 打包为
#       单输入、单 Softmax 概率输出的标准计算图, 方便导出为 ONNX / TFLite / Core ML。
# ==========================================
class MobileDeployWrapper(nn.Module):
    def __init__(self, model_list, temperatures=None, weights=None, alpha=0.5, class_bias=None, num_classes=6):
        super(MobileDeployWrapper, self).__init__()
        self.models_ = nn.ModuleList([m.eval() for m in model_list])
        n = len(model_list)
        self.register_buffer('temperatures', torch.tensor(temperatures or [1.0] * n, dtype=torch.float))
        self.register_buffer('weights', torch.tensor(weights or [1.0 / n] * n, dtype=torch.float))
        self.register_buffer('class_bias', torch.tensor(class_bias or [1.0] * num_classes, dtype=torch.float))
        self.alpha = float(alpha)

    def forward(self, x):
        probs = [F.softmax(m(x) / self.temperatures[i], dim=1) for i, m in enumerate(self.models_)]
        if len(probs) == 1:
            p = probs[0] * self.class_bias.view(1, -1)
            return p / p.sum(dim=1, keepdim=True)
        return ensemble_probs(probs, self.weights, self.alpha, self.class_bias)

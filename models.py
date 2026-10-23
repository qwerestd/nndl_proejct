import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ==========================================
# 基础组件: SENet 通道注意力模块 (论文 1 公式 2-4)
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
# 模型 1: HybridTrashNet (基于论文 1 ResSeMo 架构)
# ==========================================
class HybridTrashNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.3):
        super(HybridTrashNet, self).__init__()
        resnext = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        self.branch1_features = nn.Sequential(*list(resnext.children())[:-2]) # 2048

        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.branch2_features = mobilenet.features # 960

        self.se_block1 = SEBlock(2048, reduction=16)
        self.se_block2 = SEBlock(960, reduction=16)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2048 + 960, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Hardswish(inplace=True)
        )
        self.target_cam_layer = SEBlock(512, reduction=16) # 用于 Grad-CAM 抓取特征
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.aux_head1 = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes), nn.LogSoftmax(dim=1)
        )
        self.aux_head2 = nn.Sequential(
            nn.Linear(960, 256), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes), nn.LogSoftmax(dim=1)
        )
        self.main_classifier = nn.Sequential(
            nn.Linear(512, 256), nn.Hardswish(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes), nn.LogSoftmax(dim=1)
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
            return out_main, self.aux_head1(v1), self.aux_head2(v2)
        return out_main


# ==========================================
# 模型 2: RWCNet (还原论文 2 图 2c 跨层双流架构)
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
            nn.Linear(128, num_classes), nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.classifier(self.features(self.se(x)))


class RWCNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.3):
        super(RWCNet, self).__init__()
        # DenseNet201 三阶段拆分
        df = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT).features
        self.dense_stage1 = nn.Sequential(df.conv0, df.norm0, df.relu0, df.pool0, df.denseblock1, df.transition1, df.denseblock2) # 512
        self.dense_stage2 = nn.Sequential(df.transition2, df.denseblock3) # 1792
        self.dense_stage3 = nn.Sequential(df.transition3, df.denseblock4, df.norm5) # 1920

        # MobileNetV2 三阶段拆分
        mf = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features
        self.mobile_stage1 = nn.Sequential(*mf[:7])   # 32 (第5个倒残差块输出)
        self.mobile_stage2 = nn.Sequential(*mf[7:18]) # 320 (最后一个倒残差块输出)
        self.mobile_stage3 = mf[18]                   # 1280 (最终卷积输出)

        # 两个跨层辅助分支
        self.aux_branch1 = RWCAuxiliaryBranch(512 + 32, num_classes, dropout_rate)
        self.aux_branch2 = RWCAuxiliaryBranch(1792 + 320, num_classes, dropout_rate)

        # 最终输出层
        self.target_cam_layer = SEBlock(1920 + 1280, reduction=16) # 用于 Grad-CAM 抓取特征
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1920 + 1280, 512), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes), nn.LogSoftmax(dim=1)
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
            return out_final, out_aux2, out_aux1

        return out_final
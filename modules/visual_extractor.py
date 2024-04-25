import torch
import torch.nn as nn
import torchvision.models as models

# 用于提取图像特征的视觉提取器模块
class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor # 该字段表示要使用的视觉提取器模型的名称。
        self.pretrained = args.visual_extractor_pretrained # 该字段表示是否使用预训练的视觉提取器模型。
        # 使用 getattr 函数根据模型名称从 torchvision.models 模块中加载对应的预训练模型，
        # 根据 visual_extractor_pretrained 参数决定是否使用预训练模型
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        # 获取模型的所有子模块，并且去除了最后两个子模块（通常是全局池化层和分类器层）
        modules = list(model.children())[:-2]
        # 用剩余子模块构建一个顺序模型
        self.model = nn.Sequential(*modules)
        # 创建一个平均池化层，用于计算图像的平均特征。
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images) # 得到图像的特征表示
        # 池化+去掉维度+改变维度
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1) # 将提取的特征进行形状变换
        return patch_feats, avg_feats

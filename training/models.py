#coding:utf8
"""
包含训练中所使用的网络定义的模块。灵感来自于 https://github.com/adambielski/siamese-tr
iplet/blob/master/networks.py ，它定义了表示体系结构分支阶段的嵌入网络，并定义了将嵌入网
络作为参数的联接网络（孪生或三元组网络）。
"""

import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, zeros_, normal_

# 以下三个类是三种特征提取网络，任选一个使用即可
class BaselineEmbeddingNet(nn.Module):  # 类似 AlexNet的实现
    def __init__(self):
        super(BaselineEmbeddingNet, self).__init__()
        # 一共包含 5 个卷积模块
        self.fully_conv = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11,
                                                  stride=2, bias=True), # 第一个卷积模块，步长为 2
                                        nn.BatchNorm2d(96),
                                        nn.ReLU(),
                                        nn.MaxPool2d(3, stride=2),  # 池化层

                                        # 剩下的四个卷积模块步长都是 1，并各带一个池化
                                        nn.Conv2d(96, 256, kernel_size=5,
                                                  stride=1, groups=2,
                                                  bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.MaxPool2d(3, stride=1),
                                        nn.Conv2d(256, 384, kernel_size=3,
                                                  stride=1, groups=1,
                                                  bias=True),
                                        nn.BatchNorm2d(384),
                                        nn.ReLU(),
                                        nn.Conv2d(384, 384, kernel_size=3,
                                                  stride=1, groups=2,
                                                  bias=True),
                                        nn.BatchNorm2d(384),
                                        nn.ReLU(),
                                        nn.Conv2d(384, 32, kernel_size=3,
                                                  stride=1, groups=2,
                                                  bias=True))

    def forward(self, x):
        output = self.fully_conv(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class VGG11EmbeddingNet_5c(nn.Module):

    def __init__(self):
        super(VGG11EmbeddingNet_5c, self).__init__()
        self.fully_conv = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3,
                                                  stride=1, bias=True), # 第一个卷积，步长为 1
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, stride=2), # 最大池化步长是 2

                                        # 剩下的都是类似的
                                        nn.Conv2d(64, 128, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, stride=2),
                                        nn.Conv2d(128, 256, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        # Added ConvLayer, not in original model
                                        nn.Conv2d(256, 32, kernel_size=3,
                                                  stride=1, bias=True))

    def forward(self, x):
        output = self.fully_conv(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class VGG16EmbeddingNet_8c(nn.Module):

    def __init__(self):
        super(VGG16EmbeddingNet_8c, self).__init__()
        self.fully_conv = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, stride=2),
                                        nn.Conv2d(64, 128, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 128, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, stride=2),
                                        nn.Conv2d(128, 256, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        # Added ConvLayer, not in original model
                                        nn.Conv2d(256, 32, kernel_size=3,
                                                  stride=1, bias=True))

    def forward(self, x):
        output = self.fully_conv(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

# 以上三个类是三种特征提取网络

## 网络定义
class SiameseNet(nn.Module):
    """
    这是一个基本的孪生网络联接网络，它将两个嵌入分支的输出应用相关操作进行联接。它应该始终
    与形式为[B x C x H x W]的张量一起使用，您必须始终包括批处理维度。
    """
    # 参数：一个提取特征的子网络，上采样，输出的特征图大小，卷积的步长
    def __init__(self, embedding_net, upscale=False, corr_map_size=33, stride=4):

        """
        输入:
        embedding_net: 提取特征的子网络
        corr_map_size: 输出的特征图大小
        stride: 步长
        """

        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net  # 提取特征的子网络
        self.match_batchnorm = nn.BatchNorm2d(1)    # 通道数为 1 的 BN 层

        self.upscale = upscale ## 如果指定了上采样，
        # TODO calculate automatically the final size and stride from the
        # parameters of the branch
        self.corr_map_size = corr_map_size
        self.stride = stride
        self.upsc_size = (self.corr_map_size-1)*self.stride + 1 ## 最终经过上采样后的特征图大小

        if upscale:
            self.upscale_factor = 1
        else:
            self.upscale_factor = self.stride

    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): The reference patch of dimensions [B, C, H, W].
                Usually the shape is [8, 3, 127, 127].      孪生网络中输入的小图像
            x2 (torch.Tensor): The search region image of dimensions
                [B, C, H', W']. Usually the shape is [8, 3, 255, 255].      孪生网络中输入的大图像
        Returns:
            match_map (torch.Tensor): The score map for the pair. For the usual
                input shapes, the output shape is [8, 1, 33, 33].
        """
        embedding_reference = self.embedding_net(x1)       # 提取特征
        embedding_search = self.embedding_net(x2)          # 提取特征

        # 基于两个图像的特征来计算相关性
        match_map = self.match_corr(embedding_reference, embedding_search)
        return match_map

    def get_embedding(self, x):
        return self.embedding_net(x)

    def match_corr(self, embed_ref, embed_srch):
        b, c, h, w = embed_srch.shape
        # 采用分组的卷积来进行相关操作，参数不需要进行学习，把 batch看作卷积的输出通道数
        # 把参考图当作一个卷积核
        match_map = F.conv2d(embed_srch.view(1, b * c, h, w),
                             embed_ref, groups=b)
        # 重新排列维度，以获取批处理维度
        match_map = match_map.permute(1, 0, 2, 3)
        match_map = self.match_batchnorm(match_map)
        if self.upscale:
            match_map = F.interpolate(match_map, self.upsc_size, mode='bilinear',
                                      align_corners=False)
        return match_map

def weights_init(model):
    """
    使用 Xavier 来对卷积网络的权重进行初始化
    """
    if isinstance(model, nn.Conv2d):
        xavier_uniform_(model.weight, gain=math.sqrt(2.0))
        constant_(model.bias, 0.1)
    elif isinstance(model, nn.BatchNorm2d):
        normal_(model.weight, 1.0, 0.02)
        zeros_(model.bias)

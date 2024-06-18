#coding:utf8
import numpy as np

## 给一对训练样本，计算出标签矩阵
def create_BCELogit_loss_label(label_size, pos_thr, neg_thr, upscale_factor=4):
    """
    为了模拟逻辑损失的三级标签的行为，每个像素有两个维度，一个编码正面（1）和负面（0）像素，
    另一个编码像素是正面/负面（1）还是中性（0）

    中性值不会对梯度产生贡献，因此在计算损失时它们被忽略，它们在第一个维度上的值可以任意设置。

    阈值是根据搜索图像中的像素距离定义的，因此，如果相关性图没有进行上采样（即步长> 1），
    则必须将阈值转换为相关性图中相应的距离（通过简单地将阈值除以步长）。相关性图对应于搜索图像
    的中心子区域（通常，对于255像素的搜索图像和127像素的参考图像，该子区域有129个像素），因此
    阈值存在实际限制：任何大于此子区域的半对角线（通常为91.2像素）的值都将包含整个相关性图。

    这些测试是 dist <= pos_thr and dist > neg_thr.

    Args:
        label_size: (int) 网络输出的像素大小。
        pos_thr: (int) 正值阈值，以搜索图像的像素为单位。
        neg_thr: (int) 负值阈值，以搜索图像的像素为单位。
        upscale_factor: (int) 我们需要将输出特征图上采样多少才能匹配输入图像。它表示网络感受
        野在输入图像中的平均位移，对于相关性图中的一个像素移动。通常它告诉我们网络对图像进行了多
        少上采样或下采样。

    Returns:
        label (numpy.ndarray): 每个维度为label_size * label_size * 2的配对的标签。

    OBS: 将pos_thr和neg_thr都设置为零以获取空帧的标签。
    """

    ## pos_thr：正样本偏离的重叠阈值（原始图像空间），低于该阈值为正样本
    ## neg_thr：负样本偏离的重叠阈值（原始图像空间），高于该阈值为负样本
    pos_thr = pos_thr / upscale_factor
    neg_thr = neg_thr / upscale_factor
    # 请注意，如果标签大小为偶数，则中心可能在像素之间。
    center = (label_size - 1) / 2
    line = np.arange(0, label_size)
    line = line - center
    line = line**2
    line = np.expand_dims(line, axis=0)
    dist_map = line + line.transpose() ## 产生中心对称的标签矩阵

    '''
    array([[50, 41, 34, 29, 26, 25, 26, 29, 34, 41, 50],
           [41, 32, 25, 20, 17, 16, 17, 20, 25, 32, 41],
           [34, 25, 18, 13, 10, 9, 10, 13, 18, 25, 34],
           [29, 20, 13, 8, 5, 4, 5, 8, 13, 20, 29],
           [26, 17, 10, 5, 2, 1, 2, 5, 10, 17, 26],
           [25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25],
           [26, 17, 10, 5, 2, 1, 2, 5, 10, 17, 26],
           [29, 20, 13, 8, 5, 4, 5, 8, 13, 20, 29],
           [34, 25, 18, 13, 10, 9, 10, 13, 18, 25, 34],
           [41, 32, 25, 20, 17, 16, 17, 20, 25, 32, 41],
           [50, 41, 34, 29, 26, 25, 26, 29, 34, 41, 50]])
    '''

    ## 第 1维标志正负样本，第 2维标志有效性
    label = np.zeros([label_size, label_size, 2]).astype(np.float32)
    label[:, :, 0] = dist_map <= pos_thr**2
    ## 重叠度处于适中的样本有效性置为0，不使用该类样本
    label[:, :, 1] = (dist_map <= pos_thr**2) | (dist_map > neg_thr**2)

    return label

#coding:utf8
import numpy as np
from sklearn.metrics import roc_auc_score

# 计算中心偏移距离
def center_error(output, label, upscale_factor=4):
    """
    Args:
        output: (np.ndarray) The output of the network with dimension [Bx1xHxW]
        label: (np.ndarray) The labels with dimension [BxHxWx2] (Not used, kept
            for constistency with the metric_fcn(output,label) calls)
        upscale_factor: (int) Indicates the how much we must upscale the output
            feature map to match it to the input images. When we use an upscaling
            layer the upscale_factor is 1.

    Returns:
        c_error: (int) The center displacement in pixels.
    """
    b = output.shape[0]
    s = output.shape[-1]
    out_flat = output.reshape(b, -1)
    max_idx = np.argmax(out_flat, axis=1) ## 求最大概率值
    estim_center = np.stack([max_idx//s, max_idx % s], axis=1) ## 最大概率值位置
    dist = np.linalg.norm(estim_center - s//2, axis=1) ## 元素平方求和之后开根号
    c_error = dist.mean()
    c_error = c_error * upscale_factor ## 映射到图像空间
    return c_error

# 评估指标的计算
def AUC(output, label):
    """
    计算给定输出的ROC曲线下的面积。计算AUC时，忽略标签中性区域的所有输出（见labels.py）.
    Args:
        output: (np.ndarray) [Bx1xHxW]维度的网络输出
        label: (np.ndarray) 尺寸为[BxHxWx2]的标签
    """
    b = output.shape[0]
    # 预测概率
    output = output.reshape(b, -1)
    mask = label[:, :, :, 1].reshape(b, -1)
    # 分类标签
    label = label[:, :, :, 0].reshape(b, -1)
    total_auc = 0
    for i in range(b):
        total_auc += roc_auc_score(label[i], output[i], sample_weight=mask[i])
    return total_auc/b

# 包含所有计算指标的字典.
METRICS = {
    'AUC': {'fcn': AUC, 'kwargs': {}},
    'center_error': {'fcn': center_error,
                     'kwargs': {'upscale_factor': 4}}
}

#!/usr/bin/env python3
# coding: utf8
import cv2
from os.path import dirname, abspath, join, isfile
import numpy as np
import torch
import os
import training.models as mdl
from torchvision import transforms
from PIL import Image

## 载入模型
def load_state_keywise(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location='cpu')
    key = list(pretrained_dict.keys())[0]
    # 1. filter out unnecessary keys
    # 1.1 multi-GPU ->CPU
    if (str(key).startswith('module.')):  # 如果是多 gpu训练，那么去掉module前缀
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if
                           k[7:] in model_dict and v.size() == model_dict[k[7:]].size()}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.size() == model_dict[k].size()}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


device = torch.device("cuda") if torch.cuda.is_available \
    else torch.device("cpu")

# video_path = "test/test_video/video1.mp4" ## 视频路径
# reference_image_path = "test/test_video/video1.jpg" ## 参考图路径
# result_image_path = "test/result/video1/video1_1" ## 结果图路径
# result_video_path = "test/result/video1/result_video1_1.avi" ## 结果视频路径


reference_image_size = 127  ## 参考图尺寸
search_image_size = 255  ## 搜索图尺寸
outfeature_size = 33  ## 相关图尺寸
stride = 4  ## 全局步长
reinit_obj = 1  ## 是否更新跟踪目标

# 选择对哪个视频进行跟踪处理，更改 str 里的数字即可
video = "video1"

root_testpath = "test/test_video"  # 存放输入图片、视频的根路径
root_resultpath = "test/result"  # 存放结果的根路径
video_path = join(root_testpath, video + ".mp4")  # 视频路径
reference_image_path = join(root_resultpath, video, "reference.jpg")  # 参考图路径

second_dir_resultpath = join(root_resultpath, video)      # 存放结果路径的次级目录
# 如果没有这个文件目录，则创建一个该文件
if os.path.lexists(second_dir_resultpath) == False:
    os.makedirs(second_dir_resultpath)

result_image_path = join(root_resultpath, video, video + "_" + str(reinit_obj))  # 结果图路径

# 如果没有这个文件夹，那就先创建一个
if os.path.lexists(result_image_path) == False:
    os.makedirs(result_image_path)

result_video_path = join(root_resultpath, video, "result_" + video + "_" + str(reinit_obj) + ".avi")  # 结果视频路径

model_path = "best.pth.tar"  ## 模型路径


# 载入模型
net = mdl.SiameseNet(mdl.BaselineEmbeddingNet(), corr_map_size=outfeature_size, stride=stride)

# 载入权重
load_state_keywise(net, model_path)

data_transforms = transforms.Compose([transforms.ToTensor()])
torch.no_grad()

# 读取输入视频
cap = cv2.VideoCapture(video_path)

# 如果没有预设参考图，自动获取第一帧图像并存储
if not (os.path.exists(reference_image_path)):
    ret, frame = cap.read()
    cv2.imwrite(reference_image_path, frame)

frameIdx = 0

cv2.namedWindow('frame', 0)
x = 0
y = 0

# 读取参考图
reference_image = Image.open(reference_image_path)
# 保存原始图的尺寸
ini_weight = reference_image.width
ini_height = reference_image.height

# 缩放到参考图的尺寸
reference_image = reference_image.resize((reference_image_size, reference_image_size), Image.NEAREST)

# 预处理
reference_blob = data_transforms(reference_image).unsqueeze(0)
print("reference_blob.shape", reference_blob.shape)

# 获得帧率
fps = int(cap.get(cv2.CAP_PROP_FPS))
# 创建结果视频
videoout = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                           (ini_weight, ini_height))

while (True):
    # 每次读取一张图像
    ret, frame = cap.read()
    # 如果视频已经结束了
    if (ret == False):
        break
    # 转为 RGB 格式
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 缩放到 search_image 的大小
    search_image = cv2.resize(framergb, (search_image_size, search_image_size), interpolation=cv2.INTER_NEAREST)
    # 如果是第一帧
    if frameIdx == 0:
        print("initial")
    else:
        print("to be continued")

    # 对搜索图也进行预处理
    search_blob = data_transforms(search_image).unsqueeze(0)
    print("search_blob.shape", search_blob.shape)
    # 向网络中输入参考图和搜索图，获得结果
    out = net(reference_blob, search_blob)
    print("out.shape", out.shape)

    prob = out.squeeze().detach().numpy()
    # 以下两行进行滤波操作，提高鲁棒性
    kernel = np.ones((5, 5), np.float32) / 25
    prob = cv2.filter2D(prob, -1, kernel)

    cv2.imshow("prob", ((prob - np.min(prob)) / (np.max(prob) - np.min(prob)) * 255.0).astype(np.uint8))

    # 取概率图中最大的 index
    maxindex = np.argmax(prob)
    # 获得最大位置的 x 和 y
    y = maxindex // prob.shape[0]
    x = maxindex % prob.shape[0]
    print('center y', y)
    print('center x', x)

    # 对应原图中匹配y坐标
    srcy = stride * y
    # 对应原图中匹配x坐标
    srcx = stride * x

    cv2.imshow('reference_image', cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR))
    if frameIdx > 0:
        frame = cv2.resize(frame, (search_image_size, search_image_size), interpolation=cv2.INTER_NEAREST)
        # 画一个圆
        cv2.circle(frame, (srcx + reference_image_size // 2, srcy + reference_image_size // 2), 10,
                   (55, 255, 155), 1)
        # 画一个框x
        cv2.rectangle(frame, (srcx, srcy), (srcx + reference_image_size, srcy + reference_image_size), (0, 0, 255), 2,
                      8, 0)

        # 还原原始的尺寸
        frame_ini = cv2.resize(frame, (ini_weight, ini_height), interpolation=cv2.INTER_NEAREST)
        # 把中间每一帧都存储下来
        cv2.imwrite(os.path.join(result_image_path, str(frameIdx) + '.png'), frame_ini)
        videoout.write(frame_ini)

        # 表示更新初始的跟踪目标与否，不更新就一直用第一帧作为参考图
        if reinit_obj > 0:
            # 把框框里的图片截出来作为下一次搜索的参考图
            reference_image = np.copy(search_image[srcy:srcy + reference_image_size, srcx:srcx + reference_image_size])
            # 对下一帧的参考图进行预处理
            reference_blob = data_transforms(reference_image).unsqueeze(0)

    cv2.imshow('frame', frame)

    print("frameIdx=", frameIdx)
    frameIdx = frameIdx + 1

    # 删除这部分代码则无法实时显示跟踪情况
    k = cv2.waitKey(10)
    if k == ord('q'):
        break

videoout.release()
cap.release()
cv2.destroyAllWindows()

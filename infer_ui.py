import cv2
from os.path import join, dirname, relpath
import numpy as np
import torch
import os
import math
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import training.models as mdl
from torchvision import transforms
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap


# 载入模型
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

# 对视频进行处理的函数
def video_track(Gui):

    device = torch.device("cuda") if torch.cuda.is_available \
        else torch.device("cpu")

    gui = Gui

    model_path = "D:/desktop/graduation_design/SiameseFC/best.pth.tar"  # 模型路径
    reinit_obj = gui.sign_trackWay  # 是否更新跟踪目标
    reference_image_size = 127  # 参考图尺寸
    search_image_size = 255  # 搜索图尺寸
    outfeature_size = 33  # 相关图尺寸
    stride = 4  # 全局步长

    # 获取在 GUI 界面中所选择的视频路径
    video_path = gui.video_path
    # 获得该视频的上级路径
    root_testpath = dirname(video_path)

    # 获得视频的名字
    video_name = relpath(video_path, root_testpath)
    # 后面的 .mp4 不要
    video_name = video_name[:-4]

    # 如果所选择的视频有标注
    if (gui.sign_annot is True):
        annot_path = gui.video_path[:-4]
        xml_files_name = sorted([f for f in os.listdir(annot_path) if f.endswith(".xml")])

    # 存放结果的根路径
    root_resultpath = "D:/desktop/graduation_design/SiameseFC/test/result"

    # 存放结果路径的次级目录
    second_dir_resultpath = join(root_resultpath, video_name)

    # 如果没有这个文件目录，则创建一个该文件
    if os.path.lexists(second_dir_resultpath) == False:
        os.makedirs(second_dir_resultpath)

    # 初试参考图路径
    reference_image_path = join(second_dir_resultpath, "ini_reference.jpg")

    # 结果图路径
    result_image_path = join(second_dir_resultpath, video_name + "_" + str(reinit_obj))

    # 如果没有这个文件夹，那就先创建一个
    if os.path.lexists(result_image_path) == False:
        os.makedirs(result_image_path)

    # 结果视频路径
    result_video_path = join(second_dir_resultpath, "result_" + video_name + "_" + str(reinit_obj) + ".avi")

    # 载入模型
    net = mdl.SiameseNet(mdl.BaselineEmbeddingNet(), corr_map_size=outfeature_size, stride=stride)

    # 载入权重
    load_state_keywise(net, model_path)

    # 转换图像格式，并将多个转换方法串联起来
    data_transforms = transforms.Compose([transforms.ToTensor()])
    # 避免在评估模型时不必要的计算和存储，从而提高程序效率
    torch.no_grad()

    # 读取输入视频
    cap = cv2.VideoCapture(video_path)

    # 自动获取第一帧图像并存储到参考图路径
    ret, frame = cap.read()
    cv2.imwrite(reference_image_path, frame)

    frameIdx = 0

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
    ## 创建结果视频
    videoout = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                               (ini_weight, ini_height))

    # 画center_error坐标图用，以上一帧为蓝色，始终以第一帧为红色
    if reinit_obj == 1:
        color = 'ro-'
    else:
        color = 'bo-'

    sign_set_reference_image = 0
    num_frame = 0   # 这个视频中 frame 的数量
    while (True):
        # 每次读取一张图像
        ret, frame = cap.read()
        # 如果视频已经结束了
        if (ret == False):
            break

        # 获得初始帧的长宽信息，以及resize的倍数，以便后续计算 center_error
        weight, length, _ = frame.shape
        mul_l = length / search_image_size
        mul_w = weight / search_image_size
        # 转为 RGB 格式
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 缩放到 search_image 的大小
        search_image = cv2.resize(framergb, (search_image_size, search_image_size), interpolation=cv2.INTER_NEAREST)
        # 如果是第一帧
        if frameIdx == 0:
            print("initial---------------------------------------------------------")
        else:
            print("to be continued-------------------------------------------------")

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

        # 让其最小值为 0
        prob = prob - np.min(prob)

        # 取概率图中最大的 index
        maxindex = np.argmax(prob)
        # 获得最大位置的 x 和 y
        x = maxindex % prob.shape[0]
        y = maxindex // prob.shape[0]
        print('center y', y)
        print('center x', x)

        # 对应原图中匹配x坐标
        srcx = stride * x
        # 对应原图中匹配 y坐标
        srcy = stride * y

        # 如果选择的视频有标注，计算 center_error
        # +++++++++++++++++++++++++++++++++++++
        if(gui.sign_annot is True):
            xml_filename = xml_files_name[num_frame]
            xml_filepath = gui.video_path[:-4] + "/" + xml_filename # 得到xml文件的路径
            tree = ET.parse(xml_filepath)   # 把xml文件转成树的形式
            root = tree.getroot()           # root 是这个树的根
            obj = root.find('object')       # 找到标注中的“object”
            bbox = obj.find('bndbox')       # 找到bndbox，是这个目标的框
            xmax = int(bbox.find('xmax').text)
            xmin = int(bbox.find('xmin').text)
            ymax = int(bbox.find('ymax').text)
            ymin = int(bbox.find('ymin').text)
            annot_x = (xmax + xmin) / 4 / mul_l
            annot_y = (ymax + ymin) / 4 / mul_w
            # 计算平均中心偏移距离
            center_error = math.sqrt((annot_x - srcx) ** 2 + (annot_y - srcy) ** 2)
            if gui.sign_trackWay == 1:
                gui.center_errors_1.append(center_error)
            else:
                gui.center_errors_0.append(center_error)

            if gui.sign_trackWay == 1:
                center_errors_list = gui.center_errors_1
            else:
                center_errors_list = gui.center_errors_0

            # 绘制平均center_error的图像
            if num_frame > 0 :
                plt.plot([num_frame - 1, num_frame], [sum(center_errors_list[:-1]) /
                    len(center_errors_list[:-1]), sum(center_errors_list) / len(center_errors_list)], color)

        # ++++++++++++++++++++++++++++++++++++

        # 测试 +++++++++++++++++++++++++++++++++++++++++++++++++++
        left = right = up = down = 0
        if(x - 3 >= 0):
            x_left = x - 3
        else:
            x_left = 0
            right = 3 - x

        if(x + 3 <= 33):
            x_right = x + 3
        else:
            x_right = 33
            left = 33 - x + 3

        if(y - 3 >= 0):
            y_up = y - 3
        else:
            y_up = 0
            down = 3 - y

        if(y + 3 <= 33):
            y_down = y + 3
        else:
            y_down = 33
            up = 33 - y + 3

        prob_total = 0
        for i in range(y_up - up, y_down + down):
            for j in range(x_left - left, x_right + right):
                prob_total += prob[i][j]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++

        #--------------------让prob在label中显示
        # 归一化处理，转成 uint8 以便图像可视化
        img_prob = (prob / (np.max(prob) - np.min(prob)) * 255.0).astype(np.uint8)

        # 把二维数组转换成图像
        img_prob = cv2.cvtColor(np.array(img_prob), cv2.COLOR_GRAY2RGB)
        # 将图像的大小调整为4的倍数，并放大，以免出现图像上有分割线的情况
        h, w, _ = img_prob.shape
        new_h = (h // 4) * 8
        new_w = (w // 4) * 8
        img_prob = cv2.resize(img_prob, (new_w, new_h))
        # 将图像数据转换为 QImage 对象
        qimage_prob = QImage(img_prob.data, img_prob.shape[1], img_prob.shape[0], QImage.Format_RGB888)
        # 计算图像的缩放比例
        scale_factor = min(gui.Label_prob.width() / qimage_prob.width(), gui.Label_prob.height() / qimage_prob.height())
        # # 调用scaled()函数进行缩放
        scaled_qimage = qimage_prob.scaled(qimage_prob.width() * scale_factor, qimage_prob.height() * scale_factor)
        # 将缩放后的图像设置为label的背景
        gui.Label_prob.setPixmap(QPixmap.fromImage(scaled_qimage))
        #-----------------------------------

        if(sign_set_reference_image == 0 or reinit_obj == 1):
            sign_set_reference_image = 1
            # -----------------------------为reference_image设置界面
            # 把二维数组转换成图像
            reference_image = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2BGR)
            # 将图像的大小调整为4的倍数，并放大，以免出现图像上有分割线的情况
            h, w, _ = reference_image.shape
            new_h = (h // 4) * 4
            new_w = (w // 4) * 4
            reference_image = cv2.resize(reference_image, (new_w, new_h))
            # 将图像转换为QImage格式
            qimage_ref_image = QImage(reference_image.data, reference_image.shape[1], reference_image.shape[0],
                                      QImage.Format_BGR888)
            # 计算图像的缩放比例
            scale_factor = min(gui.Label_ref_image.width() / qimage_ref_image.width(),
                               gui.Label_ref_image.height() / qimage_ref_image.height())
            # # 调用scaled()函数进行缩放
            scaled_qimage = qimage_ref_image.scaled(qimage_ref_image.width() * scale_factor,
                                                    qimage_ref_image.height() * scale_factor)
            # 将QImage设置给label的pixmap
            gui.Label_ref_image.setPixmap(QPixmap(scaled_qimage))
            # ---------------------------------

        if frameIdx > 0:
            frame = cv2.resize(frame, (search_image_size, search_image_size), interpolation=cv2.INTER_NEAREST)
            if(prob_total >= 0):
                print("prob_total == ", prob_total)
                # 在中心画一个圆圈
                cv2.circle(frame, (srcx + reference_image_size // 2, srcy + reference_image_size // 2), 10,
                           (55, 255, 155), 1)
                # 画一个框
                cv2.rectangle(frame, (srcx, srcy), (srcx + reference_image_size, srcy + reference_image_size),
                              (0, 0, 255), 2, 8, 0)
            # 还原原始的尺寸
            frame_ini = cv2.resize(frame, (ini_weight, ini_height), interpolation=cv2.INTER_NEAREST)
            # 把中间每一帧都存储下来
            frame_path = join(result_image_path, str(frameIdx) + '.png')
            cv2.imwrite(frame_path, frame_ini)
            videoout.write(frame_ini)

            # 表示更新初始的跟踪目标与否，不更新就一直用第一帧作为参考图，或者以上一次有目标出现时为参考图
            if reinit_obj > 0 and prob_total >= 100:
                # 此时 reference_image 是一个二维数组代表的图像
                reference_image = np.copy(
                    search_image[srcy:srcy + reference_image_size, srcx:srcx + reference_image_size])
                reference_blob = data_transforms(reference_image).unsqueeze(0)


        #----------------------- 为 frame 设置界面
        # 把二维数组转换成图像
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        # 将图像的大小调整为4的倍数，并放大，以免出现图像上有分割线的情况
        h, w, _ = frame.shape
        new_h = (h // 4) * 4
        new_w = (w // 4) * 4
        frame = cv2.resize(frame, (new_w, new_h))
        # 将图像转换为QImage格式
        qimage_frame = QImage(frame.data, frame.shape[1], frame.shape[0],
                                  QImage.Format_RGB888)
        # 计算图像的缩放比例
        scale_factor = min(gui.Label_frame.width() / qimage_frame.width(),
                           gui.Label_frame.height() / qimage_frame.height())
        # # 调用scaled()函数进行缩放
        scaled_qimage = qimage_frame.scaled(qimage_frame.width() * scale_factor,
                                                qimage_frame.height() * scale_factor)
        # 将QImage设置给label的pixmap
        gui.Label_frame.setPixmap(QPixmap(scaled_qimage))
        #--------------------------

        print("frameIdx=", frameIdx)
        frameIdx = frameIdx + 1

        # 按下 'q' 键时停止视频播放
        k = cv2.waitKey(10)
        if k == ord('q'):
            break

        # 按下“终止按钮”时，停止处理视频
        if gui.sign_expire == 1:
            break

        num_frame = num_frame + 1  # 每次 + 1

    # # 如果已经存储过以某一方式跟踪的结果了，那么在原图上继续加上另一种跟踪方式的曲线
    # if os.path.lexists(second_dir_resultpath + "/center_error.png") == True:


    # 存储 center_error
    if(gui.sign_annot is True):
        # 设置坐标轴范围和标签
        plt.xlim(0, num_frame - 1)
        if len(gui.center_errors_1) == 0:
            ylim = sum(gui.center_errors_0) / len(gui.center_errors_0)
        elif len(gui.center_errors_0) == 0:
            ylim = sum(gui.center_errors_1) / len(gui.center_errors_1)
        else:
            ylim = max(sum(gui.center_errors_0) / len(gui.center_errors_0), sum(gui.center_errors_1) / len(gui.center_errors_1))
        plt.ylim(0,  ylim * 1.5)
        plt.xlabel('Index')
        plt.ylabel('Center_Error')
        plt.savefig(second_dir_resultpath + "/center_error.png")

    videoout.release()
    cap.release()
    print("\n该视频共有：", num_frame , "帧")
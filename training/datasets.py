#coding:utf8
import os
from os.path import join, relpath, isfile
from math import sqrt
import random
import glob
import json

import numpy as np
from imageio import imread
from skimage.transform import resize as imresize
# from scipy.misc import imresize
from training.replace_fun_imresize import scipy_misc_imresize as imresize
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

from training.crops_train import crop_img, resize_and_pad
from utils.exceptions import IncompatibleImagenetStructure
from training.train_utils import get_annotations, check_folder_tree
from training.labels import create_BCELogit_loss_label as BCELoss

#-------------------------------------------------------------------
# 本脚本函数完成数据读取功能，把数据集中的视频读取进来并处理成模型训练需要的格式
#-------------------------------------------------------------------

## 数据集类
class ImageNetVID(Dataset):
    """数据集格式
    Imagenet/
    ├── Annotations
    │   └── VID
    │       ├── train
    │       └── val
    └── Data
        └── VID
            ├── train
            └── val
    """

    def __init__(self, imagenet_dir, transforms=ToTensor(),
                 reference_size=127, search_size=255, final_size=33,
                 label_fcn=BCELoss, upscale_factor=4,
                 max_frame_sep=50, pos_thr=25, neg_thr=50,
                 cxt_margin=0.5, single_label=True, img_read_fcn=imread,
                 resize_fcn=imresize, metadata_file=None, save_metadata=None):
        """ Initializes the ImageNet VID dataset object.
        ## BCELoss 表示 二分类交叉熵损失
        Args:
            imagenet_dir: 根目录
            transforms: 预处理函数
            reference_size: 参考图像尺寸
            search_size: 搜索尺寸
            final_size: 网络输出尺寸
            label_fcn: 标签函数
            upscale_factor: 全局步长
            max_frame_sep: 两帧训练样本的最大间距
            pos_thr: 正样本偏离的重叠阈值（原始图像空间），低于该阈值为正样本
            neg_thr:负样本偏离的重叠阈值（原始图像空间），高于该阈值为负样本
            cxt_margin: 缩放系数
            single_label: 使用全局标签
            img_read_fcn: 读取图片函数
            resize_fcn: 缩放图片函数
            metadata_file: 元数据路径
            save_metadata: 存储元数据路径
        """

        # Do a basic check on the structure of the dataset file-tree
        # if not check_folder_tree(imagenet_dir):
        #     raise IncompatibleImagenetStructure

        self.set_root_dirs(imagenet_dir)
        self.max_frame_sep = max_frame_sep
        self.reference_size = reference_size
        self.search_size = search_size
        self.upscale_factor = upscale_factor
        self.cxt_margin = cxt_margin
        self.final_size = final_size
        self.pos_thr = pos_thr
        self.neg_thr = neg_thr
        self.transforms = transforms
        self.label_fcn = label_fcn
        if single_label:
            self.label = self.label_fcn(self.final_size, self.pos_thr,
                                        self.neg_thr, upscale_factor=self.upscale_factor)
        else:
            self.label = None
        self.img_read = img_read_fcn
        self.resize_fcn = resize_fcn
        # Create Metadata. This section is specific to the train version（把数据集创造为想要的格式）
        self.get_metadata(metadata_file, save_metadata)

    def set_root_dirs(self, root):
        self.dir_data = join(root, 'Data', 'VID', "train")
        self.dir_annot = join(root, 'Annotations', 'VID', "train")
        self.SiameseFC_root_path = os.getcwd()


    def get_scenes_dirs(self):
        glob_expression = join(self.dir_data, '*', '*')

        glob_expression = self.SiameseFC_root_path + glob_expression
        dir_data = self.SiameseFC_root_path + self.dir_data

        # 原：relative_paths = [relpath(p, self.dir_data) for p in sorted(glob.glob(glob_expression))]
        relative_paths = [relpath(p, dir_data) for p in sorted(glob.glob(glob_expression))]
        # 最后输出应该只要 val 后面的，例如：ILSVRC2015_val_00000000
        return relative_paths

    def get_metadata(self, metadata_file, save_metadata):
        """ Gets the metadata, either by loading it from a json file or by building it from scratch.
        """
        # Check if not None
        if metadata_file and isfile(metadata_file):
            # 如果有就直接载入
            with open(metadata_file) as json_file:
                mdata = json.load(json_file)
            # Check the metadata file.
            if self.check_metadata(mdata):
                print("Metadata file found. Loading its content.")
                for key, value in mdata.items():
                    setattr(self, key, value)
                return
        # If couldn't get metadata from file, build it from scratch
        mdata = self.build_metadata()
        if save_metadata is not None:
            with open(save_metadata, 'w') as outfile:
                json.dump(mdata, outfile)

    def check_metadata(self, metadata):
        if not all(key in metadata for key in ('frames', 'annotations', 'list_idx')):
            return False
        # Check if first and last frames exist in indicated paths.
        if not (isfile(metadata['frames'][0][0]) and isfile(metadata['frames'][-1][-1])):
            return False
        return True

    ## 根据序号返回视频序列名字，E.g. 'ILSVRC2015_val_00007029'
    def get_seq_name(self, seq_idx):
        return self.frames[seq_idx][0].split(os.sep)[-2]

    ## 构建元数据
    def build_metadata(self):
        """
        frames: (list) 二维数组，存储所有帧的路径，第1维是视频序号，第2维是帧序号，frames[seq][frame]
        annotations: (list) 二维数组，存储所有帧的标注信息，annotations[seq][frame]
            the annotation information can be accessed using the keywords 'xmax', 'xmin', 'ymax', 'ymin':
            Example: del_x =  annotation['xmax'] - annotation['xmin']
        list_idx: (list) 长度等于数据集中所有帧的数量，存储的是该帧对应的视频序号，list_idx[idx].
                For example, if the first sequence has 3 frames and the second one has 4, the list would be:
                [0,0,0,1,1,1,1,2,...]
        """
        frames = []
        annotations = []
        list_idx = []

        # In the training folder the tree depth is one level deeper, so that
        # a sequence folder in training would be:
        # 'ILSVRC2015_VID_train_0000/ILSVRC2015_train_00030000', while in
        # the val set it would be: 'ILSVRC2015_val_00000000'
        scenes_dirs = self.get_scenes_dirs()

        # 相对路径无法找到文件，用绝对路径
        dir_data = self.SiameseFC_root_path + self.dir_data
        dir_annot = self.SiameseFC_root_path + self.dir_annot
        for i, sequence in enumerate(scenes_dirs): ## 遍历视频序列
            seq_frames = []
            seq_annots = []
            for frame in sorted(os.listdir(join(dir_data, sequence))): ## 遍历视频下的帧
                # So far we are ignoring the frame dimensions (h, w).
                annot, h, w, valid = get_annotations(dir_annot, sequence, frame)
                if valid:
                    seq_frames.append(join(dir_data, sequence, frame))
                    seq_annots.append(annot)
                    list_idx.append(i) ## 存储视频序列标签
            frames.append(seq_frames) ## 存储一维数组，构建二维数组
            annotations.append(seq_annots) ## 存储一维数组，构建二维数组

        metadata = {'frames': frames,
                    'annotations': annotations,
                    'list_idx': list_idx}

        for key, value in metadata.items():
            setattr(self, key, value)

        return metadata

    ## 训练时需要每次取出一个图像对进行训练，相应的标签则是这两个图像对的匹配程度
    def get_pair(self, seq_idx, frame_idx=None):
        """
        Gets two frames separated by at most self.max_frame_sep frames (in both directions)

        输入参数:
        seq_idx: 视频序列的索引，从中取两张图片，两张训练图只从同一个视频中获取，检测到有标注就认为是同一个目标
        frame_idx: (int) Optional, 第一帧的起始点

        返回值:
        first_frame_idx: self.frames中参考帧的索引
        second_frame_idx: self.frames中搜索帧的索引
        """
        size = len(self.frames[seq_idx]) ## 当前视频包含的所有帧数
        if frame_idx is None:
            first_frame_idx = random.randint(0, size-1)     ## 随机取第一帧
        else:
            first_frame_idx = frame_idx

        ## 要求两帧之间的间距不超过 max_frame_sep的值
        min_frame_idx = max(0, (first_frame_idx - self.max_frame_sep))
        ## 要求两帧之间加起来不超过 size - 1（边界）
        max_frame_idx = min(size - 1, (first_frame_idx + self.max_frame_sep))
        second_frame_idx = random.randint(min_frame_idx, max_frame_idx)
        # # Guarantees that the first and second frames are not the same though
        # # it wouldn't be that much of a problem
        # if second_frame_idx == first_frame_idx:
        #     if first_frame_idx == 0:
        #         second_frame_idx += 1
        #     else:
        #         second_frame_idx -= 1
        return first_frame_idx, second_frame_idx

    ## 根据目标框的大小以及扩充的边界区域，计算搜索区域
    def ref_context_size(self, h, w):
        """
        This function defines the size of the reference region around the
        target as a function of its bounding box dimensions when the context
        margin is added.

        输入:
        h: 高
        w: 宽
        """
        margin_size = self.cxt_margin * (w + h)     ## cxt_margin（缩放系数）为 0.5
        ref_size = sqrt((w + margin_size) * (h + margin_size))
        ## 确保输出尺寸是奇数
        ref_size = (ref_size//2)*2 + 1
        return int(ref_size)

    ## 样本预处理函数
    def preprocess_sample(self, seq_idx, first_idx, second_idx):
        """
        Loads and preprocesses the inputs and output for the learning problem.
        The input consists of an reference image tensor and a search image tensor,
        the output is the corresponding label tensor. It also outputs the index of
        the sequence from which the pair was chosen as well as the ref and search
        frame indexes in said sequences.

        Args:
            输入:
            seq_idx: (int) 整个数据集中视频序列索引
            first_idx: 序列中参考帧的索引
            second_idx: 序列中搜素帧的索引

        Returns:
            out_dict: (dict) Dictionary containing all the output information
                stored with the following keys:

                'ref_frame': (torch.Tensor) The reference frame with the
                    specified size.
                'srch_frame': (torch.Tensor) The search frame with the
                    specified size.
                'label': (torch.Tensor) The label created with the specified
                    function in self.label_fcn.
                'seq_idx': (int) The index of the sequence in terms of the self.frames
                    list.
                'ref_idx': (int) The index of the reference image inside the given
                    sequence, also in terms of self.frames.
                'srch_idx': (int) The index of the search image inside the given
                    sequence, also in terms of self.frames.
        """
        reference_frame_path = self.frames[seq_idx][first_idx]
        search_frame_path = self.frames[seq_idx][second_idx]
        ref_annot = self.annotations[seq_idx][first_idx]    ## 参考图像
        srch_annot = self.annotations[seq_idx][second_idx]  ## 寻找图像

        # Get size of context region for the reference image, as the geometric
        # mean of the dimensions added with a context margin

        ## 裁剪出参考图像区域
        ref_w = (ref_annot['xmax'] - ref_annot['xmin'])/2
        ref_h = (ref_annot['ymax'] - ref_annot['ymin'])/2

        ref_ctx_size = self.ref_context_size(ref_h, ref_w)  ## 要裁剪的正方形框大小

        ref_cx = (ref_annot['xmax']+ref_annot['xmin'])/2
        ref_cy = (ref_annot['ymax']+ref_annot['ymin'])/2

        ref_frame = self.img_read(reference_frame_path)     ## 参考帧图像
        ref_frame = np.float32(ref_frame)

        ref_frame, pad_amounts_ref = crop_img(ref_frame, ref_cy, ref_cx, ref_ctx_size)
        try:    ## 如果需要填充
            ref_frame = resize_and_pad(ref_frame, self.reference_size, pad_amounts_ref,
                                       reg_s=ref_ctx_size, use_avg=True,
                                       resize_fcn=self.resize_fcn)
        # If any error occurs during the resize_and_pad function we print to
        # the terminal the path to the frame that raised such error.
        except AssertionError:
            print('Fail Ref: ', reference_frame_path)
            raise

        ## 裁剪出搜索图像区域
        srch_ctx_size = ref_ctx_size * self.search_size / self.reference_size   ## 搜索区域是参考区域的两倍
        srch_ctx_size = (srch_ctx_size//2)*2 + 1    ## 保证是奇数

        srch_cx = (srch_annot['xmax'] + srch_annot['xmin'])/2
        srch_cy = (srch_annot['ymax'] + srch_annot['ymin'])/2

        srch_frame = self.img_read(search_frame_path)
        srch_frame = np.float32(srch_frame)
        srch_frame, pad_amounts_srch = crop_img(srch_frame, srch_cy, srch_cx, srch_ctx_size)
        try:    ## 有必要就进行填充
            srch_frame = resize_and_pad(srch_frame, self.search_size, pad_amounts_srch,
                                        reg_s=srch_ctx_size, use_avg=True,
                                        resize_fcn=self.resize_fcn)
        except AssertionError:
            print('Fail Search: ', search_frame_path)
            raise

        if self.label is not None:
            label = self.label
        else:   ## 计算标签矩阵
            label = self.label_fcn(self.final_size, self.pos_thr, self.neg_thr,
                                   upscale_factor=self.upscale_factor)

        ref_frame = self.transforms(ref_frame)
        srch_frame = self.transforms(srch_frame)

        out_dict = {'ref_frame': ref_frame, 'srch_frame': srch_frame,
                    'label': label, 'seq_idx': seq_idx, 'ref_idx': first_idx,
                    'srch_idx': second_idx }
        return out_dict

    def __getitem__(self, idx):
        """ Returns the inputs and output for the learning problem. The input
        Args:
        Returns:
            ref_frame (torch.Tensor): The reference frame with the specified size.
            srch_frame (torch.Tensor): The search frame with the specified size.
            label (torch.Tensor): The label created with the specified function in self.label_fcn.
        """
        seq_idx = self.list_idx[idx] ## 取视频序号（ list_idx 存的是 “视频序列标签” ）
        first_idx, second_idx = self.get_pair(seq_idx) ## 根据视频序号获取两帧图像
        return self.preprocess_sample(seq_idx, first_idx, second_idx)

    def __len__(self):
        return len(self.list_idx)

## 验证集数据集类，里面的函数基本上都是继承上面训练集类的函数，处理步骤也差不多
class ImageNetVID_val(ImageNetVID):
    """
    The validation version of the ImageNetVID dataset inherits from the
    train version. The main difference between the two is the way they sample
    the images: while the train version samples in train time, choosing a pair
    of images for the given sequence, the val version chooses all pairs in
    initialization time and stores them internally. The val sampling follows a
    pseudo-random sequence, but since it seeds the random sequence generators,
    the choice is deterministic, so the choice is saved in a file called
    val.metadata that can be charged at initialization time.
    """

    def __init__(self, *args, **kwargs):
        """ For the complete argument description see the Parent class. The val
        version uses the original initialization but it gets a list of pairs of
        indexes consistent with the list of frames and annotations.
        """
        super().__init__(*args, **kwargs)

    def set_root_dirs(self, root):
        self.dir_data = join(root, 'Data', 'VID', "val")
        self.dir_annot = join(root, 'Annotations', 'VID', "val")
        self.SiameseFC_root_path = os.getcwd()

    def get_scenes_dirs(self):
        """Apply the right glob function to get the scene directories, relative
        to the data directory
        """

        glob_expression = join(self.dir_data, '*')
        glob_expression = self.SiameseFC_root_path + glob_expression
        dir_data = self.SiameseFC_root_path + self.dir_data

        relative_paths = [relpath(p, dir_data) for p in sorted(glob.glob(glob_expression))]

        return relative_paths

    def check_metadata(self, metadata):
        """ Checks the val metadata, first by doing the check defined in the
        parent class, but for the val dataset, since the 'frames', 'annotations'
        and 'list_idx' have the same structure for both types.Then it checks
        the specific val elements of the metadata.

        Args:
            metadata: (dict) The metadata dictionary, derived from the json
                metadata file.
        """
        if not super().check_metadata(metadata):
            return False
        if not all(key in metadata for key in ('list_pairs', 'max_frame_sep')):
            return False
        # Check if the maximum frame separation is the same as the metadata one,
        # since the choice of pairs depends on this separation parameter.
        if metadata['max_frame_sep'] != self.max_frame_sep:
            return False
        return True

    def build_metadata(self):
        metadata = super().build_metadata()
        self.list_pairs = self.build_test_pairs()
        metadata['list_pairs'] = self.list_pairs
        metadata['max_frame_sep'] = self.max_frame_sep
        return metadata

    def build_test_pairs(self):
        """ Creates the list of pairs of reference and search images in the
        val dataset. It follows the same general rules for choosing pairs as the
        train dataset, but it does it in initialization time instead of
        train/val time. Morover it seeds the random module with a fixed seed to
        get the same choice of pairs across diferent executions.
        Returns:
            list_pairs: (list) A list with all the choosen pairs. It consists
                of a list of tuples with 3 elements each: the index inside the
                sequence of the reference and of the search frame and their
                sequence index. It has one tuple for each frame in the dataset,
                though the function could be easily extended to any number of
                pairs.
        """
        # Seed random lib to get consistent values
        random.seed(100)
        list_pairs = []
        for idx in self.list_idx:
            for frame_idx in range(len(self.frames[idx])):
                list_pairs.append([idx, *super().get_pair(idx, frame_idx)])
        random.shuffle(list_pairs)
        # Reseed the random module to avoid disrupting any external use of it.
        random.seed()
        return list_pairs

    def __getitem__(self, idx):
        """ Returns the reference and search frame indexes along with the sequence
        index corresponding to the given index 'idx'. The sequence itself is fixed
        from the initialization, so the function simply returns the idx'th element
        of the list_pairs.

        Args:
            idx: (int) The index of the current iteration of validation.
        """
        list_idx, first_idx, second_idx = self.list_pairs[idx]
        return self.preprocess_sample(list_idx, first_idx, second_idx)

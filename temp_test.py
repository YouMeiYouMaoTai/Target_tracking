import cv2
from os.path import join, dirname, abspath, isfile
import numpy as np
import torch
import os
import training.models as mdl
from torchvision import transforms
from PIL import Image
import GUI.function as ui

video_path = "D:/Desktop/graduation_design/SiameseFC/test/test_video/video1.mp4"
root_testpath = os.path.dirname(video_path)
print(root_testpath)
import cv2
import numpy as np

def nothing(x):#创建滑动条
    pass
cv2.createTrackbar('mixing', 'image', 0, 255, nothing)
import cv2
import numpy as np

img1 = cv2.imread('checkerboard .png')  # 读取两张图片
print(img1.shape)
#img.shape[0]：图像的垂直尺寸（高度）
# img.shape[1]：图像的水平尺寸（宽度）
# img.shape[2]：图像的通道数(b,g,r)
#
# 在矩阵中，[0]就表示行数，[1]则表示列数。


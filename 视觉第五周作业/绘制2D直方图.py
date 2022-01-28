import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('girl.png')
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])#获取直方图数值
                                                                      #channels=[0 ，1] 因为我们需要同时处理 H 和 S 两个通道（颜色和饱和度）。
                                                                      # bins=[180 ，256]H 通道为 180，S 通道为 256。
                                                                      # range=[0 ，180 ，0 ，256]H 的取值范围在 0 到 180，S 的取值范围在 0 到 256。

plt.imshow(hist,interpolation = 'nearest')
plt.show()
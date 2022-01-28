import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('zuqiu.jpg',0)
histr=cv2.calcHist([img],[0],None,[256],[0,256])#获取直方图数值
plt.plot(histr)#绘制直方图
plt.xlim([0,256])
plt.show()
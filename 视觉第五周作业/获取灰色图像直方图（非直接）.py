import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('zuqiu.jpg',0)#以灰度形式读取一张图片

#统计图像直方图————2种方法：calcHist()和np.histogram()
hist=cv2.calcHist([img],[0],None,[256],[0,256])#参数分别为原图像（图像格式为uint8或float32且要用[]括起来）、读入图片的通道（[0]-灰度、[0],[1],[2]-B,G,R）、
                                               # mask掩模（为全图时用none)、bin数目(256、16)、像素值范围（通常为[0,256])
                                               #别忘了中括号[img],[0],None,[256],[0,256]，只有mask没有中括号
#hist,bins=np.histogram(img.ravel(),256,[0,256])#img.ravel()将图像转成一维数组，这里没有中括号。返回两个数组：hist 和 bin_edges。
                                               # 数组 hist 显示直方图的值、bin_edges 显示 bin 边缘。bin_edges 的大小总是 1+(hist 的大小)，即 length(hist)+1。


plt.plot(hist)#绘制直方图
plt.xlim([0,256])
plt.show()
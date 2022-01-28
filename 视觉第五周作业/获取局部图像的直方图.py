#要统计图像某个局部区域的直方图只需要构建一副掩模图像。将要统计的部分设置成白色，其余部分为黑色，就构成了一副掩模图像。然后把这个掩模图像传给函数就可以了
import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('zuqiu.jpg',0)
#create a mask
mask=np.zeros(img.shape[:2],np.uint8)#返回来一个给定形状和类型的用0填充的数组(全黑）。img.shape[:2]取彩色图片的长宽，如果img.shape [:3] 则取彩色图片的长、宽、通道；
# zeros(shape, dtype=float, order=‘C’)
# shape:形状
# dtype:数据类型，可选参数，默认numpy.float64
# order:可选参数，c代表与c语言类似，行优先；F代表列优先
mask[100:300,100:400]=255#黑色底图上有一块白的
masked_img=cv2.bitwise_and(img,img,mask=mask)#保留原图上白色区域的图像，其余图像为黑色
#计算带掩码和不带掩码的直方图
#检查掩码的第三个参数
hist_full=cv2.calcHist([img],[0],None,[256],[0,256])#原图[img]上None区域的直方图
hist_mask=cv2.calcHist([img],[0],mask,[256],[0,256])#原图[img]上mask区域的直方图
plt.subplot(221),plt.imshow(img,'gray')#plt.subplot(221)=plt.subplot(2，2，1)，2，2，1分别表行数，列数，索引值（图片所在位置）
plt.subplot(222),plt.imshow(mask,'gray')
plt.subplot(223),plt.imshow(masked_img,'gray')
plt.subplot(224),plt.plot(hist_full),plt.plot(hist_mask)#plt.plot()函数是matplotlib.pyplot模块下的一个函数, 用于绘制直方图
                                                        #它可以绘制点和线, 并且对其样式进行控制
plt.xlim([0,256])#plt.xlim() 显示的是x轴的作图范围
plt.show()
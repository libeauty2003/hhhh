import cv2
import numpy as np

img1 = cv2.imread('../label.jpg')
img = cv2.GaussianBlur(img1, (5, 5), 0)#将图形进行二值化，用于后面的自适应阈值二值化
                                       #高斯滤波，(5, 5)表示高斯矩阵的长与宽都是5，标准差取0，，没有此运用函数，得到的图形将有明显的噪声，白色中具有较多的黑点。
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#将图片灰度化
dst = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
              cv2.THRESH_BINARY, 11, 2)#自适应阈值化，此函数的参数分别是：灰度的原图，最大阈值，阈值计算方法，二值化的计算方法，图片中分块的大小，阈值计算方法中的常数项
kernel = np.ones((1, 1), np.uint8)# 使用一个1*1的卷积核
dst = cv2.dilate(dst, kernel, iterations=-1)# 腐蚀操作,参数分别是原图，卷积和，迭代为1.
dst = cv2.erode(dst, kernel, iterations=1)# 膨胀操作

cv2.namedWindow('window', cv2.WINDOW_NORMAL)
cv2.imshow('window',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
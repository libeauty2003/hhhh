import cv2
import numpy as np

img=cv2.imread('../red.jpg')#读取图片
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#获取hsv值
#设定红色的阈值
lower_red = np.array([156,43,46])
upper_red = np.array([180,255,255])
mask = cv2.inRange(hsv, lower_red, upper_red)#根据阈值构建掩模
res = cv2.bitwise_and(img, img, mask=mask) #对原图像和掩模进行位运算,取出长方形
img1gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)#将图片灰度化
ret, mask2 = cv2.threshold(img1gray, 10, 255, cv2.THRESH_BINARY)#对灰度图进行二值化
rows, cols= mask2.shape#获取图片的大小
M=cv2.getRotationMatrix2D((cols/2, rows/2), 251, 1.5)#对红色矩形进行旋转（旋转中心，旋转角度，旋转后的缩放因子）
dst=cv2.warpAffine(mask2, M, (cols, rows))#第三个参数是输出图像的尺寸中心
cv2.imshow('img', dst)#显示图片
cv2.waitKey(0)
cv2.destroyAllWindows()





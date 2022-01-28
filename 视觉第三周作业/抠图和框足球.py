import cv2
import numpy as np

img1 = cv2.imread('zuqiu.jpg')  # 读取两张图片
img2 = cv2.imread('xiaoaixin.png')
print(img1.shape)
print(img2.shape)
img4=img1.copy()
img5=img1.copy()
# 我想把logo放在左上角，所以我创建了ROI
rows, cols, channels = img2.shape  # 获取logo的坐标
roi = img1[0:rows, 0:cols]  # 选取背景图片上与logo一样大的一块区域
# 现在创建logo的掩码，并同时创建其相反掩码
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 创建logo图片的灰度图
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)  # 对logo灰度图进行二值化，即非黑即白（掩码）
mask_inv = cv2.bitwise_not(mask)  # 获取反掩码
# 现在将ROI中logo的区域涂黑
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
# 仅从logo图像中提取logo区域
img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
# 将logo放入ROI并修改主图像
dst = cv2.add(img1_bg, img2_fg)
img4[0:rows, 0:cols] = dst

football = img1[385:450, 150:215]  # 截取足球
img3 = football.copy()
img5[375:460, 140:225] = (0, 0, 255)  # 将足球及外轮廓用红色填充
img5[385:450, 150:215] = img3  # 将足球粘贴回去

cv2.imshow('mask',mask)
cv2.imshow('mask_inv',mask_inv)
cv2.imshow('img1_bg',img1_bg)
cv2.imshow('img2_fg',img2_fg)
cv2.imshow('dst',dst)

# cv2.imshow('hunhetupian',img4)
# cv2.imshow('hunhetupian2',img5)
cv2.waitKey(0)
cv2.destroyAllWindows()
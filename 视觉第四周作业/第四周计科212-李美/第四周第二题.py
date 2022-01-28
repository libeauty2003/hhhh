import cv2
import numpy as np

img = cv2.imread('../gun.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)# 将图片转化为HSV格式，此格式为三通道

low = np.array([0, 153, 28])# 获取图像的阈值，创建掩膜
up = np.array([80, 232, 197])
mask = cv2.inRange(hsv, low, up)#黑白图
img1 = cv2.bitwise_and(img, img, mask=mask)#输出橙色图像（目标图像）

gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)# 将图形转为灰度，用于后面自适应二值化
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#进行自适应化二值化，把图片转化为黑白模式，用于寻找图片轮廓
contours, hie = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#cv2.findContours()函数用来查找检测物体的轮廓
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]#在多个矩形轮廓中选取最适合图片的轮廓，即轮廓中最大的那个
cnt = contours[0]
M = cv2.moments(cnt)#cnt是矩形边框的点集

x, y, w, h = cv2.boundingRect(cnt)#绘制最适应轮廓的矩形，并输出矩形的信息；cnt为轮廓点集合；x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

x1= int(M["m10"] / M["m00"])#矩形中心
y1 = int(M["m01"] / M["m00"])

print("矩形中心和长宽分别为：")
print((x1,y1),w,h)

(x, y), radius = cv2.minEnclosingCircle(cnt)#绘制最小的外接圆，并输出圆的信息
center = (int(x), int(y))
radius = int(radius)
img = cv2.circle(img, center, radius, (0, 255, 0), 2)
print("圆心、半径为：")
print((x, y),radius)

# 获取四个极点
left = tuple(cnt[cnt[:, :, 0].argmin()][0])
right = tuple(cnt[cnt[:, :, 0].argmax()][0])
top = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottom = tuple(cnt[cnt[:, :, 1].argmax()][0])
print('极点参数为：')
print(left)
print(right)
print(top)
print(bottom)

cv2.namedWindow('new', cv2.WINDOW_NORMAL)
cv2.imshow('new', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
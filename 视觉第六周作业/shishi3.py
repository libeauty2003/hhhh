import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('checkerboard .png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('white.jpg',0)
template1 = cv2.imread('black.jpg',0)
w, h = template.shape[::-1]
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)#模块匹配
threshold =0.8#设置阈值
loc1 = np.where( res >= threshold)#参数为条件、x、y    当匹配的模块大于等于阈值时，记录位置   返回的是一个数组
print(loc1)
for pt in zip(*loc1[::-1]):#当满足条件时，画一个矩形   每读到一个记录的位置时，画一个矩形
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,0), 1)

w, h = template1.shape[::-1]
res = cv2.matchTemplate(img_gray,template1,cv2.TM_CCOEFF_NORMED)#模块匹配
threshold =0.8#设置阈值
loc2 = np.where( res >= threshold)#参数为判断条件、x、y，true取下，false取y   当匹配的模块大于等于阈值时，记录位置   返回的是两个数组，第一个是行信息，第二个是列信息
for pt in zip(*loc2[::-1]):#当满足条件时，画一个矩形   每读到一个记录的位置时，画一个矩形
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)

max1=0
for x in zip(*loc1[::-1]):
    x1=x[0]
    for x2 in zip(*loc1[::-1]):
        x3=x2[0]
        if abs(x1-x3)>max1:
            max1=abs(x1-x3)
print(max1)

max2=0
for x in zip(*loc2[::-1]):
    x1=x[0]
    for x2 in zip(*loc2[::-1]):
        x3=x2[0]
        if abs(x1-x3)>max2:
            max2=abs(x1-x3)
print(max2)

max3=0
for x in zip(*loc1[::-1]):
    x1,y1=x
    for x2 in zip(*loc2[::-1]):
        x3,y3=x2
        a=(x1-x3)*(x1-x3)+(y1-y3)*(y1-y3)
        if a>max3:
            max3=a
            maxx1=x1
            maxx3=x3
            maxy1=y1
            maxy3=y3
cv2.line(img_rgb,(maxx1,maxy1),(maxx3,maxy3),(255,0,0),3)



cv2.imshow('kuang hei bai zi',img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
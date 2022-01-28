import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('zuqiu.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('football.jpg',0)
w, h = template.shape[::-1]
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)#模块匹配
threshold =0.8#设置阈值
loc = np.where( res >= threshold)#参数为条件、x、y    当匹配的模块大于等于阈值时，记录位置   返回的是一个数组
for pt in zip(*loc[::-1]):#当满足条件时，画一个矩形   每读到一个记录的位置时，画一个矩形
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)

cv2.imshow('kuang zuqiu',img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
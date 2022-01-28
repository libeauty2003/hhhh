import cv2
import numpy as np

img=cv2.imread('../hhhh.png')
img1=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.namedWindow('window',cv2.WINDOW_NORMAL)
cv2.namedWindow('window',0)
cv2.resizeWindow('window',500,500)#创建一个500*500大小的窗口

cv2.imshow('window',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
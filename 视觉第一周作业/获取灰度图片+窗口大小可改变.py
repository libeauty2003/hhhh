import cv2
import numpy as np

img=cv2.imread('shu.jpg')
img1=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('img',cv2.WINDOW_NORMAL)
#cv2.namedWindow('img',cv2.0)
cv2.imshow('img',img1)
cv2.waitKey(0)
cv2.destroyWindow()
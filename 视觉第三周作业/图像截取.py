import cv2
import numpy as np

img1=cv2.imread('zuqiu.jpg')
img2=img1[385:450,150:215]
cv2.imshow('windos',img2)
cv2.waitKey(0)
cv2.destroyWindow()
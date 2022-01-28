import numpy as np
import cv2

img=cv2.imread('red.jpg',0)#以灰度图读取图片
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
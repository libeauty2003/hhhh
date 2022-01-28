import cv2
import numpy as np

img=cv2.imread('hua.png')
img[1000:0,1000:0]=(0,0,255)
cv2.namedWindow('window',cv2.WINDOW_NORMAL)
cv2.imshow('window',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
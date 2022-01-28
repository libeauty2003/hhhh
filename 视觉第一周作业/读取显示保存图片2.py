import numpy as np
import cv2

img=cv2.imread('20211130212553.png',0)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

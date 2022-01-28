import cv2
import numpy as np

img1 = cv2.imread('same1.png')
img2 = cv2.imread('same2.png')

res = cv2.addWeighted(img1, 0.4, img2, 0.6, 0)
cv2.imshow('img', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

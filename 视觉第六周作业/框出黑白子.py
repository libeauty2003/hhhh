import cv2
import numpy as np

img1 = cv2.imread('checkerboard .png')#读取两张图片
print(img1.shape)
white=img1[85:116,50:85]#找出这个位置
black=img1[85:116,85:115]
white1=white.copy()#把这个位置复制下来，生成另一张图片
black1=black.copy()
cv2.imwrite('white.jpg',white1)
cv2.imwrite('black.jpg',black1)

cv2.imshow('point',white1)
cv2.imshow('point2',black1)
cv2.waitKey(0)
cv2.destroyAllWindows()
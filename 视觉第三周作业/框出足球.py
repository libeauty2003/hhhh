import cv2
import numpy as np

img1 = cv2.imread('zuqiu.jpg')#读取两张图片
football=img1[385:450,150:215]#截取足球
img3=football.copy()
cv2.imwrite('football.jpg',img3)
img1[375:460,140:225] = (0,0,255)#将足球及外轮廓用红色填充
img1[385:450,150:215] = img3#将足球粘贴回去

cv2.imshow('kuang zu qiu',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

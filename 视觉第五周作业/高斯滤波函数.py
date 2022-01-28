import cv2

img=cv2.imread('xiaotu.png')#高斯滤波是一种线性平滑滤波，适用于消除高斯噪声，广泛应用于图像处理的减噪过程。
# 通俗的讲，高斯滤波就是对整幅图像进行加权平均的过程，每一个像素点的值，都由其本身和邻域内的其他像素值经过加权平均后得到。
# 高斯滤波的具体操作是：用一个模板（或称卷积、掩模）扫描图像中的每一个像素，用模板确定的邻域内像素的加权平均灰度值去替代模板中心像素点的值。(高于阈值的点全部用平均值代替，达到降噪目的）

blur=cv2.GaussianBlur(img,(5,5),0)#高随滤波函数，(5, 5)表示高斯矩阵的长与宽都是5，标准差取0

cv2.imshow('GaussianBlur',blur)
cv2.imshow('GaussianBlur2',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
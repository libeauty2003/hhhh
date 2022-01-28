import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('aixin.jpg',0)
f = np.fft.fft2(img)#函数 np.fft.fft2() 可以对信号进行频率转换（高频和低频，变化快和变化慢），输出结果是一个复杂的数组
#本函数的第一个参数是输入图像，要求是灰度格式。第二个参数是可选的, 决定输出数组的大小。输出数组的大小和输入图像大小一样。
# 如果输出结果比输入图像大，输入图像就需要在进行 FFT（ 快速傅里叶变换）前补0。如果输出结果比输入图像小的话，输入图像就会被切割。
fshift = np.fft.fftshift(f)#将低频像素点移到中心
magnitude_spectrum = 20*np.log(np.abs(fshift))#构建振幅图（频率图），输出图像的中心部分更白（亮），这说明低频分量更多
# ifshift=np.fft.ifftshift(fshift)#使用函数np.fft.ifftshift() 进行逆平移操作，所以现在直流分量又回到左上角了
# iif=np.ifft2(ifshift)#使用函数 np.ifft2() 进行 FFT 逆变换

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
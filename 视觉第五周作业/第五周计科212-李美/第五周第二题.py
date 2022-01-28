import numpy as np
import cv2
from matplotlib import pyplot as plt

img=cv2.imread('../house.jpg',0)
dft=cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)#实现傅里叶变换。cv2.dft(原始图像，转换标识) 这里的原始图像必须是np.float32格式。
                      # 所以，我们首先需要使用cv2.float32 ()函数将图像转换。 而转换标识的值通常为cv2.DFT_COMPLEX_OUTPUT，用来输出一个复数阵列
dft_shift=np.fft.fftshift(dft)

cows,cols=img.shape
cow,col=int(cows/2),int(cols/2)
mask=np.ones((cows,cols,2),np.uint8)
mask[:,col-15:col+15]=255
fshift=dft_shift*mask
f_ishif=np.fft.ifftshift(fshift)
img_back=cv2.idft(f_ishif)
img_back=cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
img_back=cv2.medianBlur(img_back,5)

plt.subplot(121),plt.imshow(img,cmap='gray')
plt.title('1'),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(img_back,cmap='gray')
plt.title('2'),plt.xticks([]),plt.yticks([])
plt.show()
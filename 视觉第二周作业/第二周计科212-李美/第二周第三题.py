import cv2
import numpy as np

drawing=False#停止绘画
mode=True#模式为Ture
ix,iy=-1,-1
def draw_cicle(event,x,y,flags,param):#定义回调函数
    global ix,iy,drawing,mode#定义全局变量
    if event==cv2.EVENT_LBUTTONDOWN:#按下左键时
        drawing=True#开始绘画并记录此时的x,y
        ix,iy=x,y
    elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:#滑动左键时
        if drawing==True:
            if mode==True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)#模式为true时画填充矩形
            else:
                cv2.rectangle(img,(ix,iy),(x,y),(255,255,255),5)#模式为not true时画不填充矩形
                cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 0), -1)
    elif event==cv2.EVENT_LBUTTONUP:#松开鼠标时
        drawing==False#停止绘画

img=np.zeros((512,512,3),np.uint8)#创建一个黑色图片
cv2.namedWindow("windows")#窗口命名
cv2.setMouseCallback("windows",draw_cicle)#调用函数：在windows上画图
while(1):
    cv2.imshow("windows",img)#显示画好的图
    k=cv2.waitKey(1)&0xFF
    if k==ord('1'):
            mode=mode
    if k==ord('2'):
        mode=not mode
    elif k==27:#按ESC键推出
            break

cv2.destroyAllWindows()

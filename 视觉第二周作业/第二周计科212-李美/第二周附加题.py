import cv2
import numpy as np

drawing=False
ix,iy=-1,-1

def drawing_rectangle(event,x,y,flags,param):#定义回调函数
    global  ix,iy,drawing#定义全局变量
    if event==cv2.EVENT_LBUTTONDOWN:#按下左键时
        drawing=True#开始绘画并记录此时的x,y
        ix,iy=x,y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:#滑动左键时
        cv2.rectangle(img, (ix,iy),(x,y),(100,255,200),2)#画一个矩形
        cv2.rectangle(img,(ix,iy),(x,y),(0,0,0),-1)#画第二个矩形填补至与背景颜色相同
    elif event == cv2.EVENT_LBUTTONUP:#放开鼠标时
        drawing=False#停止绘画
img=np.zeros((512,521,3),np.uint8)#生成一个黑色图片
cv2.namedWindow("windows")#窗口命名
cv2.setMouseCallback("windows",drawing_rectangle)#调用函数（将回调函数与窗口绑定在一起）
while(1):
    cv2.imshow("windows",img)#显示图片
    k = cv2.waitKey(1) & 0xFF
    if k==27:
        break

cv2.destroyAllWindows()

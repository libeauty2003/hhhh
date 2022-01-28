import cv2
import numpy as np

def nothing(x):
    pass
drawing =False#停止绘画
ix,iy=-1,-1

def draw_circle(event,x,y,flags,param):#定义回调函数
    r=cv2.getTrackbarPos("R","image")#得到滑动条R的数值
    g=cv2.getTrackbarPos("G","image")#得到滑动条G的数值
    b=cv2.getTrackbarPos("B","image")#得到滑动条B的数值
    color=(r,g,b)
    global ix,iy,drawing#定义全局变量
    if event==cv2.EVENT_RBUTTONDOWN:#按下右键时
        drawing=True#开始绘画并记录此时的x,y
        ix,iy=x,y
    elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_RBUTTON:#滑动右键时
        if drawing==True:
            cv2.rectangle(img, (ix, iy), (x, y), color, -1)#画矩形
    elif event==cv2.EVENT_RBUTTONUP:#放开右键时
         drawing=False#停止绘画
    if event == cv2.EVENT_LBUTTONDOWN:#按下左键时
        drawing = True#开始绘画并记录此时的x,y
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:#滑动左键时
        if drawing == True:
            r=int(np.sqrt((x-ix)**2+(y-iy)**2))
            cv2.circle(img, (x, y),r,color,-1)#画圆
    elif event == cv2.EVENT_LBUTTONUP:#放开左键时
         drawing = False#停止绘画

img=np.zeros((512,512,3),np.uint8)#创建一个黑色图片
img[:]=[255,255,255]#将黑色背景改为白色
cv2.namedWindow("image")#窗口命名
cv2.createTrackbar("R","image",0,255,nothing)#创建滑动条R
cv2.createTrackbar("G","image",0,255,nothing)#创建滑动条G
cv2.createTrackbar("B","image",0,255,nothing)#创建滑动条B
cv2.setMouseCallback("image",draw_circle)#调用函数：在image上画图
while(1):
    cv2.imshow("image",img)#显示画好的图
    k=cv2.waitKey(1)&0xFF
    if k==27:#按ESC键退出
        break
cv2.destroyAllWindows()

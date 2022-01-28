import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('zuqiu.jpg',0)#读取一张图片
img2 = img.copy()#复制该图片
template = cv2.imread('football.jpg',0)#读取目标图片
w, h = template.shape[::-1]#获取目标图片的长宽
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',#列表中列出了所有6种比较方法
'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
for meth in methods:#读到第几个方法时
    img = img2.copy()#读取原图
    method = eval(meth)#eval 语句用来计算存储在字符串中的有效 Python 表达式，该方法就是刚刚读到的那个
    res = cv2.matchTemplate(img,template,method)# 应用模板匹配
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)#矩阵中最值极其索引
    print(min_val, max_val, min_loc, max_loc)
    #关于匹配方法，使用不同的方法产生的结果的意义可能不太一样，有些返回的值越大表示匹配程度越好，而有些方法返回的值越小表示匹配程度越好。
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:# 如果方法为TM_SQDIFF或TM_SQDIFF_NORMED，则取最小值
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img,top_left, bottom_right, 255, 2)#画矩形，最后两个参数为颜色和粗细

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()
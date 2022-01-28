import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('zuqiu.jpg')
color=('b','g','r')#对一个列表或数组既要遍历索引又要遍历元素时#使用内置enumerrate函数会有更加直接，优美的做法#enumerate会将数组或列表组成一个索引序列。
#使我们再获取索引和索引内容的时候更加方便。
for i,col in enumerate(color):#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
                              #sequence为一个序列、迭代器或其他支持迭代对象。start为下标起始位置。i为下标0，1，2。col为b,g,r。
    histr=cv2.calcHist([img],[i],None,[256],[0,256])#获取直方图数值
    plt.plot(histr,color=col)#绘图
    plt.xlim([0,256])
plt.show()
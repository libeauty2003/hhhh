import numpy as np
import cv2
a=np.array([[2000,3,4,5],[5,67,8,9],[1,3,400,5]])#定义一个数组
print(a)
min_val,max_val,min_indx,max_indx=cv2.minMaxLoc(a)#获取最值

print(min_val,max_val,min_indx,max_indx)#由输出结果可知，这个矩阵a的最小值为1.0，索引为（0，2），最大值为67.0索引为（1，1）
#索引即数值所在位置，矩阵的位置为：
#   0 1 2 3
# 0
# 1
# 2
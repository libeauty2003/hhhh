##  学习过程中用到的代码：

学习的大致步骤：

 1.下载labelimg，并使用指令安装。
  2.网上下载图片，使用labelimg制作.xml文件。
  3.下载darknet,并尝试测试。
  4.按教程建立目录，并将文件放在正确的位置。
  5.运行有关代码，并配置相应文件。
  6.开始训练。
  7.进行测试。



学习中用到的代码：

1.修改配置文件Makefile （如果使用GPU）

​    ARCH= -gencode arch=compute_35,code=sm_35 \
​                 -gencode arch=compute_50,code=[sm_50,compute_50] \
​                 -gencode arch=compute_52,code=[sm_52,compute_52] \
​                 -gencode arch=compute_70,code=[sm_70,compute_70] \
​                 -gencode arch=compute_75,code=[sm_75,compute_75]

2.编译

   make

3.训练

   ./darknet detector train cfg/my_data.data cfg/my_yolov3.cfg darknet53.conv.74

4.测试

   ./darknet detect cfg/my_yolov3.cfg weights/my_yolov3.weights 1.jpg



5.test.py文件

import os

import random
trainval_percent = 0.1
train_percent = 0.9
xmlfilepath = 'Annotations'
txtsavepath = 'ImageSets\Main'
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
ftrainval = open('ImageSets/Main/trainval.txt', 'w')
ftest = open('ImageSets/Main/test.txt', 'w')
ftrain = open('ImageSets/Main/train.txt', 'w')
fval = open('ImageSets/Main/val.txt', 'w')
for i in list:
  name = total_xml[i][:-4] + '\n'
  if i in trainval:
    ftrainval.write(name)
    if i in train:
      ftest.write(name)
    else:
      fval.write(name)
  else:
    ftrain.write(name)
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

6.my_lables.py文件

   import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
#源代码sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007',
#'val'), ('2007', 'test')]
sets=[('myData', 'train')]#改成自己建立的myData

classes = ["bird","horse","dog"] # 改成自己的类别
def convert(size, box):
  dw = 1./(size[0])
  dh = 1./(size[1])
  x = (box[0] + box[1])/2.0 - 1
  y = (box[2] + box[3])/2.0 - 1
  w = box[1] - box[0]
  h = box[3] - box[2]
  x = x*dw
  w = w*dw
  y = y*dh
  h = h*dh
  return (x,y,w,h)
def convert_annotation(year, image_id):
  in_file = open('myData/Annotations/%s.xml'%(image_id))#源代码

#VOCdevkit/VOC%s/Annotations/%s.xml
  out_file = open('myData/labels/%s.txt'%(image_id), 'w')#源代码

#VOCdevkit/VOC%s/labels/%s.txt
  tree=ET.parse(in_file)
  root = tree.getroot()
  size = root.find('size')
  w = int(size.find('width').text)
  h = int(size.find('height').text)
  for obj in root.iter('object'):
    difficult = obj.find('difficult').text
    cls = obj.find('name').text
    if cls not in classes or int(difficult)==1:
      continue
    cls_id = classes.index(cls)
    xmlbox = obj.find('bndbox')
    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
    bb = convert((w,h), b)
    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) +
'\n')
wd = getcwd()
for year, image_set in sets:
  if not os.path.exists('myData/labels/'):#改成自己建立的myData

os.makedirs('myData/labels/')

  image_ids = open('myData/ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
  list_file = open('myData/%s_%s.txt'%(year, image_set), 'w')
  for image_id in image_ids:
    list_file.write('%s/myData/JPEGImages/%s.jpg\n'%(wd, image_id))
    convert_annotation(year, image_id)
  list_file.close()

​    



## 学习心得:

​    此次学习对我来说非常困难，学习时也是困难重重。
​    首先，拿到这次题目，我就去了解了一下labelimg，得知它是一个图片标注工具。然后就按照教程进行操作，并且遇到了第一重困难——编译darknet。因为想着便于做笔记和对windows系统比较熟悉，所以在犹豫之下，我还是选择了windows系统。结果没想到，编译时比较困难，要通过第三方程序，对linux环境进行模拟，并且还要解决windows和linux系统之间编码不同的问题，虽然第一个问题得以解决，但是编码问题较为复杂，我实在无能为力，于是我放弃了windows系统，改用linux。果然，在linux系统上就没有这个问题，很快就编译好了，并且预测试成功了。

​    接下来就是制作自己的训练集，没错，我又遇到困难了。由于当时并不知道可以在kaggle上下载训练图集，所以我当时就采用了最笨的办法，那就是一张一张在网上下……现在想想也是一把辛酸泪……而且更过分的是标注图片！因为当时标注时，打开的是文件，而不是文件夹，所以，我点不了“上一张”、“下一张”，硬生生一张一张点开来标注，标完眼睛都要瞎了！费了九牛二虎之力，终于进入了下一环节，建文件夹、存放文件、运行代码……当然，其间也并非一帆风顺，要么是代码缩进问题，要么是注释问题，到处一大推报错，可谓是“做到哪错到哪”！不过，幸好都慢慢解决了。然后，又在训练时遇到一个大问题——图片后缀不一致，以至于训练时打不开图片。我下载的图片有三种类型：.jpg ，.jpeg ，.png。但是代码只有生成.jpg的，而且我.jpg和.jpeg的图片几乎各占一半，量都很大，代码我没法改，于是又手动改图片后缀！！没错，改了一大半！改完后，训练终于不报错了，但是，又发现了一个大大大大问题——没有GPU！训了3h，就弹出来十几行！于是，果断改用云上GPU（虽然由于各种问题浪费了许多money,但幸好结果还算尽人意！）。其间，又有许多文件配置问题。原来的文件许多地方需要改动，但是又不敢轻易乱动，生怕动错动乱了给弄瘫掉（其实瘫过几次，不过还好文件有备份，可以重试）。在查阅一些资料后，文件配置终于改好了，训练也成功完成，但是最后测试时有一个小问题，那就是图片和名字对不上。经过几经周折，最终才找到了问题的根本所在，那就是要把data/coco.names中的名字改掉。

​     经过学长、同学的帮助和自己的不断试错和修改，最后成功测试，这一路对于我这种小菜鸟来说，简直难如上青天！这一段经历也将会是我视觉学习以来最难忘的！


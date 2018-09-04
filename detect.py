# -*- coding:utf-8 -*-
import caffe
import numpy as np
import os
import time
root='/home/liuyp/liu/MobileNet-Caffe/'   #根目录
deploy=root + 'test.prototxt'    #deploy文件
caffe_model=root + 'model/mobilenet_iter_2000.caffemodel'   #训练好的 caffemodel
img=root+'/testpictures/'    #随机找的一张待测图片
#img='/home/liuyp/liu/keras_mobilenet/data3/train/0/'
#labels_filename = root + 'mnist/test/labels.txt'  #类别名称文件，将数字标签转换回类别名称
list_name=['book','chopsticks','water_bottle','red_bottle','wire','wire pannel']
caffe.set_mode_gpu()
net = caffe.Net(deploy,caffe_model,caffe.TEST)   #加载model和network
accuracy=0
num=0
#图片预处理设置
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,28,28)
transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值，前面训练模型时没有减均值，这儿就不用
transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR
transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
#transformer.set_mean('data', img_mean)
transformer.set_input_scale('data', 0.017)

file_dir=os.listdir(img)
#print(file_dir)
for files in file_dir:
  im=caffe.io.load_image(img+'/'+files)                   #加载图片
  net.blobs['data'].reshape(1, 3, 224, 224)
  net.blobs['data'].data[...] = transformer.preprocess('data',im)      #执行上面设置的图片预处理操作，并将图片载入到blob中
#执行测试
  time1=time.time()
  out = net.forward()
  #prob=net.blobs['myfc7'].data[0].flatten()
  prob = out['prob']
  time2=time.time()
  print("new-----------------")
  print("detect time is ",time2-time1)
  prob = np.squeeze(prob)
  idx = np.argsort(-prob)
  print('files is ',files)
  print('idx is     ',idx)
  print('prob is',prob)
 # order=prob.argsort()[-1]
 # predict_label=list_name[order]
  predict_label=list_name[idx[0]]
  name,tp=files.split(' ')
  num+=1
  if name==predict_label:
    accuracy+=1
  #print "prob is",prob
  print('name',name)
  print('label',predict_label)
  print('acc',accuracy)
  
acc=accuracy*100/num
print("accu is",acc)
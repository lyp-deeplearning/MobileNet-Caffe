# -*- coding:utf-8 -*-
import caffe
import numpy as np
import os
import time
root='/home/liuyp/liu/MobileNet-Caffe/'   #��Ŀ¼
deploy=root + 'test.prototxt'    #deploy�ļ�
caffe_model=root + 'model/mobilenet_iter_2000.caffemodel'   #ѵ���õ� caffemodel
img=root+'/testpictures/'    #����ҵ�һ�Ŵ���ͼƬ
#img='/home/liuyp/liu/keras_mobilenet/data3/train/0/'
#labels_filename = root + 'mnist/test/labels.txt'  #��������ļ��������ֱ�ǩת�����������
list_name=['book','chopsticks','water_bottle','red_bottle','wire','wire pannel']
caffe.set_mode_gpu()
net = caffe.Net(deploy,caffe_model,caffe.TEST)   #����model��network
accuracy=0
num=0
#ͼƬԤ��������
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #�趨ͼƬ��shape��ʽ(1,3,28,28)
transformer.set_transpose('data', (2,0,1))    #�ı�ά�ȵ�˳����ԭʼͼƬ(28,28,3)��Ϊ(3,28,28)
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #��ȥ��ֵ��ǰ��ѵ��ģ��ʱû�м���ֵ������Ͳ���
transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR
transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
#transformer.set_mean('data', img_mean)
transformer.set_input_scale('data', 0.017)

file_dir=os.listdir(img)
#print(file_dir)
for files in file_dir:
  im=caffe.io.load_image(img+'/'+files)                   #����ͼƬ
  net.blobs['data'].reshape(1, 3, 224, 224)
  net.blobs['data'].data[...] = transformer.preprocess('data',im)      #ִ���������õ�ͼƬԤ�������������ͼƬ���뵽blob��
#ִ�в���
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
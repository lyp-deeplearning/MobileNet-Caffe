#encoding=utf8
from __future__ import print_function
import argparse
import os

import numpy as np
import caffe


def parse_args():
    parser = argparse.ArgumentParser(
        description='evaluate pretrained mobilenet models')
    parser.add_argument('--proto', dest='proto',
                        help="path to deploy prototxt.", type=str)
    parser.add_argument('--model', dest='model',
                        help='path to pretrained weights', type=str)
    parser.add_argument('--image', dest='image',
                        help='path to color image', type=str)
    parser.add_argument('--directory', dest='directory', default="",
                        help='path to color image directory', type=str)

    args = parser.parse_args()
    return args, parser


global args, parser
args, parser = parse_args()
label_names=['book','chopsticks','water_bottle','red_bottle','wire','wire pannel']

def eval():
    nh, nw = 224, 224
    img_mean = np.array([112.24, 115.46, 117.84], dtype=np.float32)

    caffe.set_mode_cpu()
    net = caffe.Net(args.proto, args.model, caffe.TEST)

    if not args.directory:
        im = caffe.io.load_image(args.image)
        h, w, _ = im.shape
        if h < w:
            off = (w - h) / 2
            im = im[:, off:off + h]
        else:
            off = (h - w) / 2
            im = im[off:off + h, :]
        im = caffe.io.resize_image(im, [nh, nw])

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))  # row to col
        transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR
        transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
        transformer.set_mean('data', img_mean)
        transformer.set_input_scale('data', 0.017)

        net.blobs['data'].reshape(1, 3, nh, nw)
        net.blobs['data'].data[...] = transformer.preprocess('data', im)
        out = net.forward()
        prob = out['prob']
        prob = np.squeeze(prob)
        idx = np.argsort(-prob)

        label_names = np.loadtxt('synset.txt', str, delimiter='\t')
        for i in range(5):
            label = idx[i]
            print('%.2f - %s' % (prob[label], label_names[label]))

    # Evaluate a directory
    else:
        label_names=['book','chopsticks','water_bottle','red_bottle','wire','wire pannel']
        correct = 0
        total = 0
        for f in os.listdir(args.directory):
            cls = f.split(' ')[0]
            im = caffe.io.load_image(os.path.join(args.directory, f))
            h, w, _ = im.shape
            if h < w:
                off = (w - h) / 2
                im = im[:, off:off + h]  # 
            else:
                off = (h - w) / 2
                im = im[off:off + h, :]
            im = caffe.io.resize_image(im, [nh, nw])

            transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
            transformer.set_transpose('data', (2, 0, 1))  # row to col
            transformer.set_channel_swap('data', (2, 1, 0))  # RGB to BGR
            transformer.set_raw_scale('data', 255)  # [0,1] to [0,255]
            transformer.set_mean('data', img_mean)
            transformer.set_input_scale('data', 0.017)

            net.blobs['data'].reshape(1, 3, nh, nw)
            net.blobs['data'].data[...] = transformer.preprocess('data', im)
            out = net.forward()
            prob = out['prob']
            prob = np.squeeze(prob)
            idx = np.argsort(-prob)
            
            #label_names = np.loadtxt('msynset.txt', str, delimiter='\t')
            label = label_names[idx[0]]
            
            print(idx, '--------------------------')
            print(prob, '############################')
            print('name', cls)
            print('label', label)
            print('acc', correct)
            if cls == label:
                correct += 1
            total += 1
        print('total: %d' % total)
        print('correct: %d' % correct)
        print('precision: %f' % (correct * 1. / total))

if __name__ == '__main__':
    eval()
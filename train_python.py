import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('/home/liuyp/liu/MobileNet-Caffe/solver.prototxt')
solver.net.copy_from('/home/liuyp/liu/MobileNet-Caffe/mobilenet.caffemodel')
#solver.net.copy_from('/home/liuyp/liu/MobileNet-Caffe/model/mobilenet_iter_900.caffemodel')
solver.solve()
import caffe

model_def ="/usr/work/data/deepcontext_o/network_conv_sketch_no_bn.prototxt" 
#weights = "/usr/work/data/deepcontext_o/model_finetune_iter_4000.caffemodel"
weights = "/usr/work/data/deepcontext_o/model_finetune_iter_2000.caffemodel"
net = caffe.Net(model_def, weights, caffe.TEST)

interval = 135
n_samples = 6750
niter = n_samples/interval

correct = 0
net_sum = 0
for it in range(niter):
  net.forward()
  print 'Iteration', it, 'testing...'
  correct += sum(net.blobs['fc8-n'].data.argmax(1)
                         == net.blobs['label'].data)
  net_sum += net.blobs['accuracy'].data

test_acc = correct / 6750.
net_acc = net_sum / float(niter)

print "test ", test_acc
print "net ", net_acc


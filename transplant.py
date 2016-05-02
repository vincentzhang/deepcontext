import caffe

# Load the original network and extract the fully connected layers' parameters.
old_model_def = "/usr/work/data/deepcontext_o/train_out/network_no_bn.prototxt"
net = caffe.Net(old_model_def,
                '/usr/work/data/deepcontext_o/train_out/model_no_bn_iter_6000.caffemodel',
             caffe.TEST)
params = ['fc6']
# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for fc in params:
  print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

print 'filters'
for layer_name, param in net.params.iteritems():
  print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
print 'activations'
for layer_name, blob in net.blobs.iteritems():
  print layer_name + '\t' + str(blob.data.shape)


# This is the conv net model with separate label dummy data layer
#model_conv_def = "/usr/work/data/deepcontext_o/train_out/network_conv_no_bn.prototxt"
# This is the conv net model reading from lmdb
model_conv_def = "/usr/work/data/deepcontext_o/train_out/network_conv_sketch_no_bn.prototxt"
net_full_conv = caffe.Net(model_conv_def,
                '/usr/work/data/deepcontext_o/train_out/model_no_bn_iter_6000.caffemodel',
                caffe.TEST)

params_full_conv = ['fc6-conv']
# conv_params = {name: (weights, biases)}
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)
print 'filters'
for layer_name, param in net_full_conv.params.iteritems():
  print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
print 'activations'
for layer_name, blob in net_full_conv.blobs.iteritems():
  print layer_name + '\t' + str(blob.data.shape)

# Converting weights for fc6-conv
for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]

# Save the transformed weights file
print 'Save the transformed weights file'
#conv_weight_file = '/usr/work/data/deepcontext_o/train_out/model_conv_no_bn_iter_6000.caffemodel'
conv_weight_file = '/usr/work/data/deepcontext_o/train_out/model_conv_sketch_no_bn_iter_6000.caffemodel'
net_full_conv.save(conv_weight_file)

# Be careful with this since LMDB does not allow multiple read connections
#print 'Loading model from this caffemodel'
#net_full_conv1 = caffe.Net(model_conv_def,conv_weight_file,caffe.TEST)

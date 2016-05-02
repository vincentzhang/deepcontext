from caffe import draw
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from PIL import Image
import pydot
# Save image as png
rankdir = 'BT'
net_model = caffe_pb2.NetParameter()
#model_def = "/usr/work/data/deepcontext_o/train_out/network_clean.prototxt"
#model_def = "/home/vzhang/repo/caffe/models/bvlc_reference_caffenet/deploy.prototxt"
#model_def = "/usr/work/data/deepcontext_o/train_out/network_conv_sketch_no_bn.prototxt"
model_def = "/usr/work/data/deepcontext_o/network_conv_sketch_no_bn.prototxt"
text_format.Merge(open(model_def).read(), net_model)
#print net_model
filename = 'sketch_net_conv_test.png'
draw.draw_net_to_file(net_model, filename, rankdir)
#image = Image.open(filename)
#plt.imshow(image)
#plt.show()
#image.close()

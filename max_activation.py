""" Run after running train.py and in the same ipython session """
imgdir = 'img/'
if not os.path.exists(imgdir):
  os.mkdir(imgdir)
max_pat = np.empty((9,4096,)+d.shape[2:]) # 9x4096x96x96
max_number = np.zeros((9, 4096)) # it keeps track of top 9 response in the history
max_imgid = np.zeros((9, 4096)) # it keeps track of top 9 response in the history
max_batchid = np.zeros((9, 4096)) # it keeps track of top 9 response in the history
curr_max_num = np.zeros((9, 4096)) # it keeps track of top 9 response in the history

n_batch_max = 100
for idx_batch in range(n_batch_max):
  sys.stdout.write('.')
  # load the patches data from disk
  val_data = np.load(val_data_names[idx_batch%len(val_data_names)],mmap_mode='r')
  val_perm = np.load(val_perm_names[idx_batch%len(val_perm_names)],mmap_mode='r')
  val_label = np.load(val_label_names[idx_batch%len(val_label_names)],mmap_mode='r')

  # input the patch data
  solver.net.blobs['data'].reshape(*val_data.shape)
  solver.net.blobs['data'].data[:]=val_data[:]

  # input the patch pairings
  solver.net.blobs['select1'].reshape(*(val_perm.shape[0],))
  solver.net.blobs['select1'].data[:]=val_perm[:,0]
  #print 'select1 shape: ', solver.net.blobs['select1'].data.shape
  solver.net.blobs['select2'].reshape(*(val_perm.shape[0],))
  solver.net.blobs['select2'].data[:]=val_perm[:,1]

  # input the labels
  solver.net.blobs['label'].reshape(*val_label.shape)
  solver.net.blobs['label'].data[:]=val_label

  solver.net.forward()

  # look into fc6 to find max activations
  max_idx = solver.net.blobs['fc6'].data.argsort(0)[::-1][:9] # 9x4096
  # update the image id
  curr_fc6_data = solver.net.blobs['fc6'].data  # 506x4096
  curr_max_num = np.zeros((9,4096))
  for col in range(curr_fc6_data.shape[1]):
    # curr_max-num is 9x4096
    curr_max_num[:,col] = curr_fc6_data[max_idx[:,col], col]
    # compare element-wise with each in the history, if curr num is bigger than
    # before, set it to True
    comp_idx = curr_max_num[:,col] > max_number[:,col] # boolean matrix, 9x1
    # update the maximum number
    max_number[:,col] = np.maximum(curr_max_num[:,col], max_number[:,col])
    # Use the logical idx to update the bigger entries in img index and batch
    # index
    max_imgid[comp_idx,col] = max_idx[comp_idx,col]
    max_batchid[comp_idx,col] = [idx_batch]*np.count_nonzero(comp_idx)

# Now we have two matrices, max_imgid for image index, and max_batchid for
# batch index
#def ind2sub(array_shape, ind):
#  """ Returns (r,c), the row and column index"""
#  # row major: ind = r*cols + c
#  rows = (ind.astype('int') / array_shape[1])
#  cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
#  return (rows, cols)

print "Now sorting the idx and load the patches"
idx = max_batchid.argsort(None) # as flatten array
last_batch_id = -1
for i in idx:
  sys.stdout.write('.')
  # load the patches data from disk
  # convert index to r,c subscript
  r,c = np.unravel_index(i,max_batchid.shape)
  idx_batch = int(max_batchid[r,c])
  if idx_batch != last_batch_id: 
    val_data = np.load(val_data_names[idx_batch%len(val_data_names)],mmap_mode='r')
    #val_perm = np.load(val_perm_names[idx_batch%len(val_perm_names)],mmap_mode='r')
    #val_label = np.load(val_label_names[idx_batch%len(val_label_names)],mmap_mode='r')
    last_batch_id = idx_batch
  # load image patches
  #r,c = ind2sub((9,4096),idx)
  max_pat[r,c,:,:] = val_data[max_imgid[r,c],0,:,:]/255 # this will be 96x96 patch, converted to [0,1] range

# Visualize 
print "Visualizing the patches in 3x3 grid for each unit"
## 64x64, each is 3x3
# too big of a picture
# save every 256 of the 3x3 grid into a picture
# 16 pictures in total
# data is 9x4096x96x96
# split into 9x256x96x96
def vis_activation(data):
  # data is 9x256x96x96
  m = 3
  n = int(np.sqrt(data.shape[1])) # 16
  padding = ((0,0),(0,0),(0,1),(0,1))
  data = np.pad(data, padding, mode='constant', constant_values=0)
  data = data.transpose(1,0,2,3) # 256x9x97x97
  #print "shape of data 1: ", data.shape
  data = data.reshape((n,n)+data.shape[1:]) # 16x16x9x97x97
  data = data.reshape((n,n,m,m)+data.shape[-2:]) # 16x16x3x3x96x96
  #print "shape of data 2: ", data.shape
  data = data.transpose(0,1,2,4,3,5) # 16,16,3,96,3,96
  data = data.reshape((n,n,m*data.shape[-1],m*data.shape[-1]))# 16,16,3x96,3x96
  #print "shape of data 3: ", data.shape
  padding = ((0,0),(0,0),(0,25),(0,25))
  data = np.pad(data, padding, mode='constant', constant_values=0)
  data = data.transpose(0,2,1,3)
  #print "shape of data 4: ", data.shape
  data = data.reshape((n*data.shape[-1],n*data.shape[-1]))#16x3x96,16x3x96
  plt.imshow(data, cmap=plt.cm.gray)
  plt.axis('off')
#plt.show()
# split into 9x256x96x96
div = 16
num_units = 256
for div_idx in range(div):
  vis_activation(max_pat[:,int(div_idx*num_units):int((div_idx+1)*num_units),:,:])
  plt.savefig(imgdir+'max_activation_'+str(div_idx)+'.png', bbox_inches='tight',dpi=1200)
#plt.imshow(max_pat[:,0,:,:].reshape(3,3,96,96).transpose(0,2,1,3).reshape(3*96,3*96),cmap=plt.cm.gray)

## store the top 9 activations patches for each unit, reshape, plot as 3x3 grid
#def vis_square(data):
#    data = (data - data.min()) / (data.max() - data.min())
#    n = int(np.ceil(np.sqrt(data.shape[0])))
#    #print 'number of filters is ', n
#    padding = (((0, n ** 2 - data.shape[0]),
#               (0, 1), (0, 1))
#               + ((0, 0),) * (data.ndim - 3))
#    #print 'padding is ', padding
#    #print "data shape before padding", data.shape
#    data = np.pad(data, padding, mode='constant', constant_values=1)
#    #print "data shape after padding", data.shape
#    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
#    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
#    #print "data shape", data.shape
#    plt.imshow(data,cmap=plt.cm.gray)
#    plt.axis('off')
#
#vis_square(solver.net.params['conv1'][0].data.transpose(0,2,3,1))
#plt.show()

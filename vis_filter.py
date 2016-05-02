def vis_square(data):
  data = (data - data.min()) / (data.max() - data.min())
  n = int(np.ceil(np.sqrt(data.shape[0])))
  #print 'number of filters is ', n
  padding = (((0, n ** 2 - data.shape[0]),
             (0, 1), (0, 1))
             + ((0, 0),) * (data.ndim - 3))
#    #print 'padding is ', padding
#    #print "data shape before padding", data.shape
  data = np.pad(data, padding, mode='constant', constant_values=1)
  print "data shape after padding", data.shape
  #data = data[:,:,:,0]
  data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
  data = data.reshape((n * data.shape[1], n * data.shape[3]) + \
  data.shape[4:])
#    #print "data shape", data.shape
  plt.imshow(data,cmap=plt.cm.gray)
  plt.axis('off')
#
vis_square(solver.net.params['conv1'][0].data.transpose(0,2,3,1))
plt.show()


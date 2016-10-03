import sys
#sys.path.insert(0,'caffe_ext/python');
import utils as ut
from multiprocessing import Process, Queue, Pool
import matplotlib.pyplot as plt
import numpy as np
import os, subprocess, time, math, random, shutil, signal, glob, re
from caffe import layers as L, params as P, to_proto, NetSpec, get_solver, Net
from caffe.proto import caffe_pb2
import caffe
import deepcontext_config
import pdb

# Basic Configuration
patch_sz=(96,96) # size of sampled patches
batch_sz=512     # max patches in a single batch
#batch_sz=100     # max patches in a single batch
gap = 48         # gap between patches
noise = 7        # jitter by at most this many pixels
patch_eps = 0.04

def netset(n, nm, l):
  setattr(n, nm, l);
  return getattr(n,nm);

def conv_relu(n, bottom, name, ks, nout, stride=1, pad=0, group=1, 
              batchnorm=False, weight_filler=dict(type='xavier')):
    conv = netset(n, 'conv'+name, L.Convolution(bottom, kernel_size=ks, stride=stride, 
                         num_output=nout, pad=pad, group=group, 
                         weight_filler=weight_filler))
    convbatch=conv;
    if batchnorm:
      batchnorm = netset(n, 'bn'+name, L.BatchNorm(conv, in_place=True, 
                           param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}]));
      convbatch = batchnorm
    # Note that we don't have a scale/shift afterward, which is different from
    # the original Batch Normalization layer.  Using a scale/shift layer lets
    # the network completely silence the activations in a given layer, which
    # is exactly the behavior that we need to prevent early on.
    relu=netset(n, 'relu'+name, L.ReLU(convbatch, in_place=True))
    return conv, relu 

def fc_relu(n, bottom, name, nout, batchnorm=False):
    fc = netset(n, 'fc'+name, L.InnerProduct(bottom, num_output=nout, 
                        weight_filler = dict(type='xavier')))
    fcbatch=fc;
    if batchnorm:
      batchnorm = netset(n, 'bn'+name, L.BatchNorm(fc, in_place=True,
                           param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}]));
      fcbatch = batchnorm
    relu = netset(n, 'relu'+name, L.ReLU(fcbatch, in_place=True));
    return fc, relu

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet_stack(data, n, use_bn=True):
    conv_relu(n, data, '1', 11, 96, stride=4, pad=5)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    conv_relu(n, n.norm1, '2', 5, 256, pad=2)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    conv_relu(n, n.norm2, '3', 3, 384, pad=1, batchnorm=use_bn)
    conv_relu(n, n.relu3, '4', 3, 384, pad=1, batchnorm=use_bn)
    conv_relu(n, n.relu4, '5', 3, 256, pad=1, batchnorm=use_bn)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    fc_relu(n, n.pool5, '6', 4096, batchnorm=use_bn)

def gen_net(batch_size=512, use_bn=True):
    n=NetSpec();
    n.data = L.DummyData(shape={"dim":[batch_size,3,96,96]})
    n.select1 = L.DummyData(shape={"dim":[2]})
    n.select2 = L.DummyData(shape={"dim":[2]})
    n.label = L.DummyData(shape={"dim":[2]})
    caffenet_stack(n.data, n, use_bn)
    n.first = L.BatchReindex(n.relu6, n.select1)
    n.second = L.BatchReindex(n.relu6, n.select2)
    n.fc6_concat=L.Concat(n.first, n.second);

    fc_relu(n, n.fc6_concat, '7', 4096, batchnorm=use_bn);
    fc_relu(n, n.relu7, '8', 4096);
    n.fc9 = L.InnerProduct(n.relu8, num_output=8,
                            weight_filler=dict(type='xavier'));
    n.loss = L.SoftmaxWithLoss(n.fc9, n.label, loss_param=dict(normalization=P.Loss.NONE));

    prot=n.to_proto()
    prot.debug_info=True
    return prot;

# image preprocessing.  Note that the input image will modified.
def prep_image(im):
  """
    Input: height, width, channel 
    Output:channel, height, width, 
  """
  return im.transpose(2, 0, 1)
  # for some patches, randomly downsample to as little as 100 total pixels
#  if(random.random() < .33):
#    origsz=im.shape
#    randpix=int(math.sqrt(random.random() * (95 * 95 - 10 * 10) + 10 * 10))
#    im=caffe.io.resize(im.astype(np.uint8), (randpix, randpix))
#    im=(caffe.io.resize(im, (origsz[0], origsz[1])) * 255).astype(np.float32)

  # randomly drop all but one color channel
#  chantokeep=random.randint(0, 2);
#  mean=[123, 117, 104]
#  for i in range(0, 3):
#    if i==chantokeep:
#      im[:,:,i]-=np.mean(im[:,:,i])
#    else:
#      im[:,:,i]=np.random.uniform(0, 1, (im.shape[0], im.shape[1])) - .5

  # Normalize the mean and variance so that gradients are a less useful cue;
  # then scale by 50 so that the variance is roughly the same as the usual
  # AlexNet inputs.
  #im=im / np.sqrt(np.mean(np.square(im))) * 50
  #im = im / np.sqrt(np.mean(np.square(im)))

def sample_patch(x, y, gridstartx, gridstarty, patch_sz, gap, noisehalf, 
                 im_shape):
  """
  Sample a patch.  
  This function defines a grid over the image of patches of size patch_sz 
  with a gap of 'gap' between the patches.  The upper-left corner of the grid 
  starts at (gridstartx, gridstarty).  We then sample the patch at location 
  (x, y) on this grid, jitter by up to noisehalf in every direction.  
  im_shape is the dimensions of the image; 
  an error is thrown if (x, y) refers to a patch outside the image frame. 

  Returns the coordinates of the sampled patch's upper left corner.
  """
  xpix = gridstartx + x * (patch_sz[1] + gap) \
         + random.randint(-noisehalf, noisehalf)
  xpix2 = min(max(xpix, 0), im_shape[1] - patch_sz[1])
  ypix = gridstarty + y * (patch_sz[0] + gap) \
         + random.randint(-noisehalf, noisehalf)
  ypix2 = min(max(ypix, 0), im_shape[0] - patch_sz[0])
  assert abs(xpix - xpix2) < gap
  assert abs(ypix - ypix2) < gap
  return (xpix2 , ypix2)

# A background thread which loads images, extracts pairs of patches, arranges
# them into batches.
#
# args:
#   dataq: A queue object where the batches of data will be sent.  Batches
#   consist of a tuple of: 
#     (1) datafnm: 
#         a 4-d array of of N patches, or 
#         a filename of a file containing the data,
#     (2) perm: a list of pairs of patches to be used as training examples, 
#         kept in an array of shape N-by-2,
#     (3) label: an array of labels for each pair, which are in the range 0-7.
#   batch_sz: max number of patches go in each batch
#   imgs: a list of images (see load_imageset)
#   tmpdir: if not None, data batches will be saved here and the resulting
#     filename will be sent through the queue rather than the data. 
#   seed: a seed for the random number generator, required so that different
#     processes produce images in a different order:
#   tid: thread id for use in filenames.
#   patch_sz: size of sampled patches
def imgloader(dataq, batch_sz, imgs, tmpdir, seed, tid, patch_sz): 
  qctr = 0 # idx for the batch
  curidx = 0 # idx for the image
  # order for going through the images
  np.random.seed(seed)
  imgsord=np.random.permutation(len(imgs['filename']))
  # sample this many grids per image 
  num_grids = 4
  # This is the index of the grid: 
  # randomly sample a number of times from the same image
  gridctr = 0
  # storage for the sampled batch
  perm = [] # perm[:,0]: patch 1, perm[:,1]: patch 2?
  label = [] # label
  pats = []  # patches img data, will be stored in fnm
  # index within the current batch
  j = 0;
  #tmp_count = 0
  # keep returning batches forever.  Each iteration of this loop
  # samples one grid of patches from an image.
  while True:
    # if we've already sampled num_grids in this image, we sample a new one.
    if(gridctr==0):
      while True:
        try:
          # resize each image to between 150k(387x387)->450k(670x670)
          # 1232k(1110x1110) pixels
          im=ut.get_resized_image(imgsord[curidx % len(imgs['filename'])],
                             imgs,
                             {"gri_targpixels":random.randint(150000,450000)})
        except:
          print("broken image id " + str(curidx))
          curidx = (curidx + 1) % (len(imgsord))
          continue
        curidx = (curidx + 1) % (len(imgsord))
        if(im.shape[0] > patch_sz[0] * 2 + gap + noise and 
           im.shape[1] > patch_sz[1] * 2 + gap + noise):
          # loop until find an image that's bigger than 2*patch + gap + noise
          break
      gridctr = num_grids;
    # compute where the grid starts, and then comptue its size.
    # grid start from top left corner
    gridstartx = random.randint(0, patch_sz[1] + gap - 1)
    gridstarty = random.randint(0, patch_sz[0] + gap - 1)
    # what are gridszx, y
    gridszx = int((im.shape[1] + gap-gridstartx) / (patch_sz[1] + gap))
    gridszy = int((im.shape[0] + gap-gridstarty) / (patch_sz[0] + gap))
    # Whenever we sample and store a patch, we'll put its index in this
    # variable so it's easy to pair it up later.
    # intialize to -1, fill with index j when found a match, fill with -2 
    # when it's a good patch but not yet matched with another
    grid=np.zeros((gridszy, gridszx), int)-1
    # grid for unmatched patch, initialized to -1
    # fill with j_pat when it's a good patch
    # j_pat index the saved position of the patch
    grid_u_pat=np.zeros((gridszy, gridszx), int)-1 

    # if we can't fit the current grid into the batch without going over
    # batch_sz, put the batch in the queue and reset.
    if(gridszx * gridszy + j >= batch_sz):
      #print "DEBUGGING: gridsz:(%d,%d), j:%d, len(label):%d, len(pats):%d" %\
      #(gridszx,gridszy,j,len(label), len(pats))
      pats=map(prep_image, pats)
      data=np.array(pats)
      qctr+=1
      perm=(np.array(perm))
      #print "DEBUGGING: perm length %s" %str(len(perm))
      #print "DEBUGGING: perm shape %s" %str(perm.shape)
      #print "DEBUGGING: perm type %s" %str(type(perm))
      label=(np.array(label))
      #print "DEBUGGING: len of label %s" %str(len(label))
      #print "DEBUGGING: shape of label %s" %str(label.shape)
      #print "perm", perm
      #print "label", label
      if tmpdir is None:
        dataq.put((np.ascontiguousarray(data), perm, label), timeout=600)
      else:
        # store the data in tmpdir
        fnm=tmpdir + str(tid) + '_' + str(qctr) + '.npy'
        np.save(fnm, data)
        dataq.put((fnm, perm, label), timeout=600)
      perm=[]
      label=[]
      pats=[]
      j=0

    gridctr-=1;
    # for each location in the grid, sample a patch, search up and to the
    # left for patches that can be paired with it, and add them to the batch.
    # j denotes the position of the grid, from 0 -> 8
    # for grid layout, see the comment for function pos2lbl()
    # y is the rows, x is the columns

    #initial_j = j # store j before starting checking the grid
    # used for storing all the good patches whether or not are
    # matched, format: [(ypix0,xpix0),(ypix1,xpix1),...]
    unmatched_pats=[]
    j_pat = 0
    for y in range(0,gridszy):
      for x in range(0,gridszx):
        # get the top left corner location of one patch
        (xpix, ypix)=sample_patch(x, y, gridstartx, gridstarty, patch_sz, 
                                  gap, noise, im.shape)
        # Append the patch to pats list 
        # range [0,1] -> [0,255]
        # Append the patch only when there's enough black pixels, threshold is
        # defined in `patch_eps' global variable
        tmp_patch = im[ypix:ypix + patch_sz[0], xpix:xpix + patch_sz[1], 0]
        ratio = np.count_nonzero(tmp_patch<1)/float(tmp_patch.size)
        #print ('The ratio at j %d,x %d,y %d,gridctr %d is: \
        #       %f')%(j,x,y,gridctr,ratio)
        if ratio > patch_eps: # at least 4% 
          grid[y, x] = -2 # -2 means good patch but unmatched
          grid_u_pat[y, x] = j_pat # store this patch first
          unmatched_pats.append((ypix,xpix))
          j_pat+=1
          # flag to see if the current patch has already been stored
          stored = 0
          for pair in [(-1,-1), (0,-1), (1,-1), (-1,0)]:
            gridposx = pair[0] + x;
            gridposy = pair[1] + y;
            # find the patch that gives both gridpos x and y >=0
            if(gridposx < 0 or gridposy < 0 or gridposx >= gridszx):
              continue
            # now that the current patch satisfies the condition
            # check if the matched patch also satisfies ( grid value == -1
            # means that it has not been saved before)
            if grid[gridposy, gridposx] == -1:
              continue
            # At this point, we know we have got a valid match
            # First store current patch as patch j
            if stored == 0:
              # append the current patch if not yet appended
              pats.append(np.copy(
                im[ypix:ypix + patch_sz[0], xpix:xpix + patch_sz[1], :]*255));
              stored = 1
              # Update the grid
              grid[y,x] = j
              initial_j = j # idx of the current patch
              # update j since we just added one patch
              j+=1
            if grid[gridposy, gridposx] == -2:
              # if the previous patch is unmatched,
              # assign j to it
              grid[gridposy, gridposx] = j
              j += 1
              # Update the saved pats with the real j
              # patch_pos : (y,x)
              pat_posy,pat_posx = unmatched_pats[grid_u_pat[gridposy, gridposx]]
              # append the previous patch
              pats.append(np.copy(im[pat_posy:pat_posy+patch_sz[0],
                                     pat_posx:pat_posx+patch_sz[1],:]*255))
            # two possibilites: grid=-2 or grid=some j
            perm.append(np.array([initial_j, grid[gridposy, gridposx]]))
            # this label is the position of the patch wrt the 2nd one
            label.append(pos2lbl(pair))
            perm.append(np.array([grid[gridposy, gridposx],initial_j]))
            # this label is the position of the 2nd patch wrt the 1st one
            label.append(pos2lbl((-pair[0],-pair[1])))
        #else:
          # does not contain enough strokes
        #  pass
          #grid[y, x] = -1
          #grid_u_pat[y, x] = -1
    #print "DEBUGGING: gridsz:(%d,%d), j:%d, len(label):%d, len(pats):%d" %\
    #  (gridszy,gridszx,j,len(label), len(pats))
    #print "tmp_count: %d, len(label): %d" %(tmp_count,len(label))
    #if tmp_count == 0 and len(label) > 0:
    #  fnmx1 = str(tid) + '_' + str(qctr) + '_1'  +  '.npy'
    #  fnmx2 = str(tid) + '_' + str(qctr) + '_2' +   '.npy'
    #  fnmx3 = str(tid) + '_' + str(qctr) + '_3' +  '.npy'
    #  print "writing temp result to npy file: ", fnmx1
    #  pats2=map(prep_image, pats)
    #  tmp_data=np.array(pats2)
    #  print "tmp_data type %s, %s: " %(type(tmp_data), tmp_data.shape)
      #tmp_result = (tmp_data, perm, label)
      #print "tmp_result type %s: " %(type(tmp_result))
    #  tmp_count = 1
    #  np.save(fnmx1, tmp_data)
    #  np.save(fnmx2, np.array(perm))
    #  np.save(fnmx3, np.array(label))

# convert an (x, y) offset into a single number to use as a label. Labels are:
# 1 2 3   y == -1   0 1 2
# 4   5   y == 0    3   4
# 6 7 8   y == 1    5 6 7
def pos2lbl(pos):
  """
    pos: a tuple of (x,y) 
  """
  (posx, posy)=pos;
  if(posy==-1):
    lbl = posx + 1;
  elif(posy == 0):
    lbl = (posx + 7) / 2
  else:
    assert(posy == 1);
    lbl = posx + 6;
  return lbl;

# will set these later, need to make it global for signal handler
if 'exp_name' not in locals():
  exp_name='';
def signal_handler(signal, frame):
    print("PYCAFFE IS NOT GUARANTEED TO RETURN CONTROL TO PYTHON WHEN " +
        "INTERRUPTED. That means I can't necessarily clean up temporary files " +
        "and spawned processes. " +
        "You were lucky this time. Run deepcontext_quit() to quit. Next time touch " + 
        exp_name + '_pause to pause safely and ' + exp_name + '_quit to quit.')

def deepcontext_quit():
  if 'prevOutFd' in locals():
    os.dup2(prevOutFd, sys.stdout.fileno())
    os.close(prevOutFd)
    os.dup2(prevErrFd, sys.stderr.fileno())
    os.close(prevErrFd)
  for proc in procs:
    proc.terminate()
  time.sleep(2)
  shutil.rmtree(tmpdir)
  os.kill(os.getpid(), signal.SIGKILL)

def load_imageset():
  """
    return a dict containing two fields 'dir' (a string) and 'filename' (a list 
    of strings) such that
    scipy.misc.imread(imgs['dir'] + imgs['filename'][idx]) will return
    an image.  
    Make sure the order of imgs['filename'] is deterministic, since
    the code uses the index in this list as an ID for each image.
  """
  datadir=deepcontext_config.imagenet_dir
  imgs={}
  imgs['dir']=datadir+'train/'
  names=[]
  with open(datadir + 'train.txt', 'rb') as f:
    for line in f:
      names.append(line.rstrip()) # strip the newline character
      #row=line.split()
      #names.append(row[0])
  imgs['filename']=names
  return imgs

# The main code body.  
if __name__ == '__main__':
  try:
    if 'solver' not in locals():
      # This returns the current filename without extention
      exp_name=ut.mfilename();
      # all generated files will be here.
      outdir = deepcontext_config.out_dir + '/' + exp_name + '_out/';
      if deepcontext_config.tmp_dir:
        tmpdir = deepcontext_config.tmp_dir + '/' + exp_name + '_out/';
      else:
        tmpdir = None
      if not os.path.exists(outdir):
        os.mkdir(outdir);
      else:
        try:
          input=raw_input;
        except:
          pass;
        print('=======================================================================');
        print('Found old data. Load most recent snapshot and append to log file (y/N)?');
        inp=input('======================================================================');
        if not 'y' == inp.lower():
          raise RuntimeError("User stopped execution");
          
      if not os.path.exists(tmpdir):
        os.makedirs(tmpdir);
      # by default, we append to the logfile if it's already there.
      #if os.path.exists(outdir + "out.log"):
      #  os.remove(outdir + "out.log")

    # Magic commands to redirect standard output and standard
    # error to a log file for easy plotting of the loss function.  Note that
    # running these commands will screw up your terminal; hence why the whole
    # code is wrapped in a try/finally statement that puts things back the way
    # they were.
    prevOutFd = os.dup(sys.stdout.fileno())
    prevErrFd = os.dup(sys.stderr.fileno()) 
    tee = subprocess.Popen(["tee", "-a", outdir + "out.log"], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

    # if the solver hasn't been set up yet, do so now.  Otherwise assume that 
    # we're continuing.
    if 'solver' not in locals():
      if os.path.isfile(exp_name + '_pause'):
        print('Removing the pause file')
        os.remove(exp_name + '_pause')
      if os.path.isfile(exp_name + '_quit'):
        print('Removing the quit file')
        os.remove(exp_name + '_quit')

      with open(outdir+'network.prototxt','w') as f:
        f.write(str(gen_net()));
      with open(outdir+'network_no_bn.prototxt','w') as f:
        f.write(str(gen_net(use_bn=False)));

      ut.mkorender('solver_mko.prototxt', outdir + 'solver.prototxt', 
                   base_lr=1e-5, outdir=outdir, weight_decay=0, momentum=0.9)

      print('setting gpu')
      caffe.set_mode_gpu();
      print('constructing solver')
      solver = caffe.get_solver(outdir + 'solver.prototxt');

      # Find earlier solvers and restore them
      fils=glob.glob(outdir + 'model_iter_*.solverstate');
      if(len(fils)>0):
        idxs=[];
        for f in fils:
          idxs.append(int(re.findall('\d+',os.path.basename(f))[0]));
        # Load the latest solver state
        idx=np.argmax(np.array(idxs))
        solver.restore(outdir + os.path.basename(fils[idx]));

      # we occasionally read out the parameters in this list and save the norm
      # of the update out to disk, so we can make sure they're updating at
      # the right speed.
      track=[]
      for bl in solver.net.params:
        if not 'bn' in bl:
          print "bl: ", bl
          track.append(bl)
      nrm=dict()
      # validation ?
      intval={};
      trackold={};
      for tracknm in track:
        intval[tracknm]=[]
        nrm[tracknm]=[]
      print 'intval', intval
      print 'nrm', nrm
      curstep = 16000

      # load the images
      imgs=load_imageset()

      # start the data prefetching threads.
      dataq=[]
      procs=[]
      i=0
      for i in range(0,3):
          # Start 3 image loader
        dataq.append(Queue(5))
        procs.append(Process(target=imgloader, 
                         args=(dataq[-1], batch_sz, imgs, tmpdir,
                               (hash(outdir)+i) % 1000000, #random seed
                               i, patch_sz)))
        procs[-1].start()
      signal.signal(signal.SIGINT, signal_handler)

      # Total number of training iterations
      niter = 12000

      # Set up test data before starting training
      test_interval = 100
      test_niter = niter/test_interval # test only for 20 times
      n_test_batch = 100
      # losses will also be stored in the log
      # probably better to save it to logs to prevent memory over use
      #train_loss = np.zeros(test_niter*test_interval)
      #test_acc = np.zeros(test_niter)

      ## Read validation data for testing
      valdir = deepcontext_config.out_dir + '/val_data/'
      valfile = deepcontext_config.out_dir + '/val_filelist.txt'
      val_result_file = outdir + 'val_result'
      # Grab a list of file names
      # read images, save to val set
      with open(valfile, 'r') as f:
        val_data_names = [valdir + line.rstrip() for line in f]
      val_label_names = []
      val_perm_names = []
      for val_data_name in val_data_names:
        # val_data_name is the file name of data file
        val_label_names.append(val_data_name.split('.npy')[0] + '_label.npy')
        val_perm_names.append(val_data_name.split('.npy')[0] + '_perm.npy')

    ## The main loop over batches.
    while True:#curstep < niter:
      # Training 
      start=time.time()
      (datafnm, perm, label)=dataq[curstep % len(dataq)].get(timeout=600)
      print("queue size: " + str(dataq[curstep % len(dataq)].qsize()))

      if(tmpdir is None):
        d=datafnm
      else:
        # load the patches data from disk
        d=np.load(datafnm,mmap_mode='r')
        os.remove(datafnm)

      # input the patch data
      solver.net.blobs['data'].reshape(*d.shape)
      solver.net.blobs['data'].data[:]=d[:]
      #print 'data shape: ', solver.net.blobs['data'].data.shape

      # input the patch pairings
      solver.net.blobs['select1'].reshape(*(perm.shape[0],))
      solver.net.blobs['select1'].data[:]=perm[:,0]
      #print 'select1 shape: ', solver.net.blobs['select1'].data.shape
      solver.net.blobs['select2'].reshape(*(perm.shape[0],))
      solver.net.blobs['select2'].data[:]=perm[:,1]
      #print 'select2 shape: ', solver.net.blobs['select2'].data.shape

      # input the labels
      solver.net.blobs['label'].reshape(*label.shape)
      solver.net.blobs['label'].data[:]=label

      print("data input time: " + str(time.time()-start));

      # take a step
      solver.step(1)
      print("norm_loss: " + str(solver.net.blobs['loss'].data /
            (label.shape[0])));
      print("solver step time: " + str(time.time() - start));
          # store the train loss
      #train_loss[curstep] = solver.net.blobs['loss'].data/ label.shape[0]

      ### Testing
      ### Test at every interval
      if curstep % test_interval == 0:
        # run a full test every so often
        print 'Iteration', curstep, '. Overall iter: ', solver.iter , 'testing...'
        correct = 0
        len_label = 0 # this is the total length of labels over entire batch
        total_loss = 0
        start=time.time()
        # test over the first 100 batches in the list
        for idx_batch in range(n_test_batch):
          sys.stdout.write('.')
          # load the patches data from disk
          val_data = np.load(val_data_names[idx_batch%len(val_data_names)],mmap_mode='r')
          val_perm = np.load(val_perm_names[idx_batch%len(val_perm_names)],mmap_mode='r')
          val_label = np.load(val_label_names[idx_batch%len(val_label_names)],mmap_mode='r')
          #plt.imshow(d[0,:,:,:].transpose(1,2,0)/255,cmap=plt.cm.gray)

          # input the patch data
          solver.net.blobs['data'].reshape(*val_data.shape)
          solver.net.blobs['data'].data[:]=val_data[:]
          #print 'data shape: ', solver.net.blobs['data'].data.shape

          # input the patch pairings
          solver.net.blobs['select1'].reshape(*(val_perm.shape[0],))
          solver.net.blobs['select1'].data[:]=val_perm[:,0]
          #print 'select1 shape: ', solver.net.blobs['select1'].data.shape
          solver.net.blobs['select2'].reshape(*(val_perm.shape[0],))
          solver.net.blobs['select2'].data[:]=val_perm[:,1]
          #print 'select2 shape: ', solver.net.blobs['select2'].data.shape

          # input the labels
          solver.net.blobs['label'].reshape(*val_label.shape)
          solver.net.blobs['label'].data[:]=val_label

          solver.net.forward()
          correct += sum(solver.net.blobs['fc9'].data.argmax(1)
                        == solver.net.blobs['label'].data)
          len_label += val_label.shape[0]
          total_loss +=solver.net.blobs['loss'].data/(val_label.shape[0])
 
        #test_acc[curstep // test_interval] = correct / float(len_label)
        print("Validation batch time: " + str(time.time() - start))
        print "Validation accuracy: ", correct / float(len_label)
        #  test_acc[curstep // test_interval]
        print "Validation batch norm_loss: " , total_loss/n_test_batch
    
        # plot after all test iterations finished
        #print "Plotting the test accuracy and train loss"
        #_, ax1 = plt.subplots()
        #ax2 = ax1.twinx()
        #ax1.plot(np.arange(test_niter*test_interval), train_loss)
        #ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
        #ax1.set_xlabel('iteration')
        #ax1.set_ylabel('train loss')
        #ax2.set_ylabel('test accuracy')
        #ax2.set_title('Custom Test Accuracy: {:.2f}'.format(test_acc[-1]))
        #plt.savefig(outdir+'test_inter_acc_'+str(test_interval)+'_'+str(test_niter)+'.png', bbox_inches='tight')

      dobreak=False
      broken=[]

      msg = (' Please examine the situation and re-execute ' + exp_name + 
             '.py to continue.')
      # Peek into the network every 100 iterations
      if curstep % 100 == 0:
        start = time.time()
        print("getting param statistics...")
        for tracknm in track:
          try:
            intval[tracknm].append(np.sqrt(np.sum(np.square(
                solver.net.params[tracknm][0].data - trackold[tracknm]))));
            if (intval[tracknm][-1] > 10 * intval[tracknm][-2] 
                and curstep > 100) \
                or np.isnan(intval[tracknm][-1]):
              print(tracknm + " changed a suspiciously large amount." + msg)
              dobreak = True
              broken.append(tracknm)
          except:
            print("init " + tracknm + " statistics")
          trackold[tracknm]=np.copy(solver.net.params[tracknm][0].data)
          nrmval=np.sqrt(np.sum(np.square(solver.net.params[tracknm][0].data)))
          nrm[tracknm].append(nrmval)
        np.save(outdir + 'intval',intval)
        np.save(outdir + 'nrm',nrm)
        print("param statistics time: " + str(time.time()-start));

        # Check fc8 weights
        val = np.sum(solver.net.params["fc8"][0].data)
        if np.isnan(val) or val > 1e10:
          print("fc8 activations look broken to me." + msg)
          dobreak = True
        # Check the gradients of pool1 activations.
        # broken if the gradients are too big
        val2 = np.max(np.abs(solver.net.blobs["pool1"].diff));
        if np.isnan(val2) or val2 > 1e8:
          print("pool1 diffs look broken to me." + msg)
          dobreak = True

      curstep += 1
      if dobreak:
        break
      if os.path.isfile(exp_name + '_pause'):
        break
      if os.path.isfile(exp_name + '_quit'):
        # Need to kill the subprocesses and delete the temporary files.
        deepcontext_quit()
    # Store the testing results
    #val_result = {'train_loss': train_loss, 'test_acc':test_acc}
    #np.save(val_result_file, val_result)

  except KeyboardInterrupt:
    if 'procs' in locals():
      handler(None,None)
      deepcontext_quit()
    #raise
  finally:
    if 'procs' in locals():
      for proc in procs:
        proc.terminate()
        time.sleep(2)
      shutil.rmtree(tmpdir)
    if 'prevOutFd' in locals():
      os.dup2(prevOutFd, sys.stdout.fileno())
      os.close(prevOutFd)
      os.dup2(prevErrFd, sys.stderr.fileno())
      os.close(prevErrFd)

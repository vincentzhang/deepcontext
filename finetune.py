import sys
import utils as ut
from multiprocessing import Process, Queue, Pool
import matplotlib.pyplot as plt
import numpy as np
import os, subprocess, time, math, random, shutil, signal, glob, re
from caffe import layers as L, params as P, to_proto, NetSpec, get_solver, Net
from caffe.proto import caffe_pb2
import caffe
import deepcontext_config

# will set these later, need to make it global for signal handler
if 'exp_name' not in locals():
  exp_name=''
def signal_handler(signal, frame):
    print("PYCAFFE IS NOT GUARANTEED TO RETURN CONTROL TO PYTHON WHEN " +
        "INTERRUPTED. That means I can't necessarily clean up temporary files " +
        "and spawned processes. " +
        "You were lucky this time. Run deepcontext_quit() to quit. Next time touch " + 
        exp_name + '_pause to pause safely and ' + exp_name + '_quit to quit.')

def deepcontext_quit():
  print "Entered deepcontext_quit"
  #if 'prevOutFd' in locals():
  if True:
    print "Entered deepcontext_quit, start clean up"
    os.dup2(prevOutFd, sys.stdout.fileno())
    os.close(prevOutFd)
    os.dup2(prevErrFd, sys.stderr.fileno())
    os.close(prevErrFd)
    os.kill(os.getpid(), signal.SIGKILL)

def vis_square(data):
  data = (data - data.min()) / (data.max() - data.min())
  n = int(np.ceil(np.sqrt(data.shape[0])))
  print 'number of filters is ', n
  padding = (((0, n ** 2 - data.shape[0]),
             (0, 1), (0, 1))
             + ((0, 0),) * (data.ndim - 3))
  #print 'padding is ', padding
  #print "data shape before padding", data.shape
  data = np.pad(data, padding, mode='constant', constant_values=1)
  print "data shape after padding", data.shape
  #data = data[:,:,:,0]
  data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
  data = data.reshape((n * data.shape[1], n * data.shape[3]) + \
  data.shape[4:])
  #print "data shape", data.shape
  #plt.imshow(data,cmap=plt.cm.gray)
  plt.imshow(data)
  plt.axis('off')
#
if __name__ == '__main__':
  if 'solver' not in locals():
    # This returns the current filename without extention
    exp_name=ut.mfilename()
    # all generated files will be here.
    outdir = deepcontext_config.out_dir + '/' + exp_name + '_out/'
    if not os.path.exists(outdir):
      os.mkdir(outdir)
    else:
      try:
        input=raw_input
      except:
        pass
      print('=======================================================================');
      print('Found old data. Load most recent snapshot and append to log file (y/N)?');
      inp=input('======================================================================');
      if not 'y' == inp.lower():
        raise RuntimeError("User stopped execution")

  # redirect stdout,stderr
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

    # msra filler
    #ut.mkorender('solver_finetune_mko.prototxt', outdir + 'finetune_solver.prototxt', 
    #             base_lr=1e-3, outdir=deepcontext_config.out_dir,
    #             weight_decay=0, momentum=0.9,
    #             fnm='network_conv_sketch_no_bn_no_freeze_msra_filler.prototxt',solver_type='SGD')
                 #fnm='network_conv_sketch_no_bn_no_freeze.prototxt',solver_type='SGD')
    # random init
    #             fnm='network_conv_sketch_no_bn_no_freeze_random_weight.prototxt',solver_type='SGD')
    ut.mkorender('solver_finetune_mko.prototxt', outdir + 'finetune_solver.prototxt', 
                 base_lr=1, outdir=deepcontext_config.out_dir,
                weight_decay=0.0005, momentum=0.9,
                fnm='network_conv_sketch_no_bn_no_freeze.prototxt',solver_type='AdaDelta')
    #ut.mkorender('solver_finetune_mko.prototxt', outdir + 'finetune_solver.prototxt', 
    #             base_lr=1e-2, outdir=deepcontext_config.out_dir, weight_decay=0, momentum=0.9,
    #              fnm='simple.prototxt')

    print('setting gpu')
    caffe.set_mode_gpu()
    print('constructing solver')
    solver = caffe.get_solver(outdir + 'finetune_solver.prototxt')
    #conv_weight_file = deepcontext_config.out_dir + '/model_conv_sketch_no_bn_iter_6000.caffemodel'
    # ultimate fine-tuning, magic init
    # conv_weight_file = deepcontext_config.out_dir + '/outputmodel.caffemodel'
    # one conv6
    #conv_weight_file = deepcontext_config.out_dir + '/magic/sketch_no_bn_one_conv6.caffemodel'
    # fc9 
    #conv_weight_file = deepcontext_config.out_dir + '/magic/sketch_no_bn_one_conv6.caffemodel'
    # 2000 iterations
    #conv_weight_file = deepcontext_config.out_dir + '/model_finetune_iter_2000.caffemodel'
    #solver.net.copy_from(conv_weight_file)

    # Find earlier solvers and restore them
#    fils=glob.glob(outdir + 'model_finetune_iter_*.solverstate');
#    if(len(fils)>0):
#      idxs=[];
#      for f in fils:
#        idxs.append(int(re.findall('\d+',os.path.basename(f))[0]));
      # Load the latest solver state
#      idx=np.argmax(np.array(idxs))
#      solver.restore(outdir + os.path.basename(fils[idx]));


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
    intval={}
    trackold={}
    for tracknm in track:
      intval[tracknm]=[]
      nrm[tracknm]=[]
    print 'intval', intval
    print 'nrm', nrm


    layer_names = solver.net.params.keys()
    # Set up test data before starting training
#    test_interval = 100
#    test_niter = niter/test_interval
#    n_test = 1350

    for curstep in xrange(12000):
      dobreak=False
      broken=[]
      msg = (' Please examine the situation and re-execute ' + exp_name + 
             '.py to continue.')

      solver.step(1)

      # Print the accuracy
      # equivalent to the auto message if display=1 in the solver prototxt file
      #acc = sum(solver.net.blobs['label'].data == \
      #    solver.net.blobs['fc8-n'].data.argmax(1))/float(solver.net.blobs['label'].data.shape[0])
      #print "training accuracy at iter ", curstep, ' is ', acc

      # testing
#      if curstep % test_interval == 0:
#        solver.test_nets[0]

      if curstep % 100 == 0:
        # Plot histogram for debugging
        # conv1, conv2, conv3, conv4, conv5, fc6-conv, fc6b-conv, fc7-n, fc8-n
        for layer_name in layer_names:
          feat = solver.net.blobs[layer_name].data[0]
          _ = plt.hist(feat.flatten(), bins=20)
          plt.xlabel('Activation') 
          plt.ylabel('Count')
          plt.title('Activation Histogram of layer '+ layer_name) 
          plt.savefig(outdir + layer_name + '_step_' + str(curstep) + '_hist.png',bbox_inches='tight')
          plt.close() # close this figure to prevent memory leak

        vis_square(solver.net.params['conv1'][0].data.transpose(0,2,3,1))
        plt.savefig(outdir + 'step_' + str(curstep) + '_conv1_filter.png',bbox_inches='tight')
        plt.close()

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

        # Check fc7-n weights
        #val = np.sum(solver.net.params["fc7-n"][0].data)
        #if np.isnan(val) or val > 1e10:
        #  print("fc7-n activations look broken to me." + msg)
        #  dobreak = True
        # Check the gradients of pool1 activations.
        # broken if the gradients are too big
        val2 = np.max(np.abs(solver.net.blobs["pool1"].diff));
        if np.isnan(val2) or val2 > 1e8:
          print("pool1 diffs look broken to me." + msg)
          dobreak = True
        # log activation means, check if activations are 0
        print 'activations mean of all layers'
        for layer_name, blob in solver.net.blobs.iteritems():
          print layer_name + '\t' + str(blob.data.mean())
        print "filter weights sum"
        for layer_name, param in solver.net.params.iteritems():
          print layer_name + ' weights: \t' + str(param[0].data.sum())

      if dobreak:
        break
      if os.path.isfile(exp_name + '_pause'):
        break
      if os.path.isfile(exp_name + '_quit'):
        deepcontext_quit()

  if 'prevOutFd' in locals():
    os.dup2(prevOutFd, sys.stdout.fileno())
    os.close(prevOutFd)
    os.dup2(prevErrFd, sys.stderr.fileno())
    os.close(prevErrFd)


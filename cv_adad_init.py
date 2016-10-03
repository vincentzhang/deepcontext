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
  os.dup2(prevOutFd, sys.stdout.fileno())
  os.close(prevOutFd)
  os.dup2(prevErrFd, sys.stderr.fileno())
  os.close(prevErrFd)

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
    cv_step = 0
    force_quit = False

    while True:
      if force_quit:
        break
      # Hyperparams
      delta_val = 10 ** np.random.uniform(-8,-6.9)
      #weight_decay_val = 10 ** np.random.uniform(-4,-1)
      weight_decay_val = 0.0005
      #momentum_val_list = [0.9, 0.95]
      #momentum_val = momentum_val_list[np.random.randint(0,2)]
      momentum_val = 0.9
      # Print out information
      print "CV step %d, Start with parameter" % cv_step
      print "Delta = %.2e" % delta_val
      print "Regularization = %.2e" % weight_decay_val # 0.0005?
      print "Momentum = %.2f" % momentum_val

      # msra filler
      solver_file = outdir + 'hyperparam_cv_solver.prototxt'
      ut.mkorender('solver_cv_adad_mko.prototxt', solver_file,
                   base_lr=1,
                   outdir=deepcontext_config.out_dir,delta=delta_val,
                   weight_decay=weight_decay_val, momentum=momentum_val,
                   fnm='network_conv_sketch_no_bn_no_freeze_msra_filler.prototxt',
                   solver_type='AdaDelta')

      print('setting gpu')
      caffe.set_mode_gpu()
      print('constructing solver')
      solver = caffe.get_solver(solver_file)

      # we occasionally read out the parameters in this list and save the norm
      # of the update out to disk, so we can make sure they're updating at
      # the right speed.
      track=[]
      for bl in solver.net.params:
        if not 'bn' in bl:
          #print "bl: ", bl
          track.append(bl)
      nrm=dict()
      # validation ?
      intval={}
      trackold={}
      for tracknm in track:
        intval[tracknm]=[]
        nrm[tracknm]=[]
      #print 'intval', intval
      #print 'nrm', nrm

      # save mean value of the last 20 losses
      loss_history = []
      loss_len = 0
      prev_loss_mean = 87
      layer_names = solver.net.params.keys()
      num_iter = 105 * 5
      for curstep in xrange(num_iter+1):
        dobreak=False
        broken=[]
        msg = (' Please examine the situation and re-execute ' + exp_name + 
               '.py to continue.')

        solver.step(1)
        # Check if loss has been too big
        loss = solver.net.blobs['loss'].data
        if curstep >= 50 and loss > 87:
          print "Loss is %f" % loss
          print "Network exploded, exiting..."
          dobreak = True
        # ***************************************************************
        # This part monitors the training loss and stops training if it
        # plateaued
        #loss_history.append(loss)
        #loss_len+=1
        #if loss_len == 50:
        #  loss_mean = np.mean(loss_history)
        #  eps =  0.05
        #  # change has to be bigger than 0.05 of previous average
        #  if np.divide(abs(loss_mean - prev_loss_mean),prev_loss_mean) < eps:
        #    # exit cuz it's plateaued
        #    dobreak = True
        #    print "Stop this CV param set, since training loss only improved \
        #    %.2f in the last 100 iterations\n" \
        #      % np.divide(abs(loss_mean - prev_loss_mean), prev_loss_mean)
        #  else:
        #    prev_loss_mean = loss_mean
        #    loss_history = []
        #    loss_len=0
        # ***************************************************************
        if curstep % num_iter == 0: # 1 epoch
          # Plot histogram for debugging
          # conv1, conv2, conv3, conv4, conv5, fc6-conv, fc6b-conv, fc7-n, fc8-n
          for layer_name in layer_names:
            feat = solver.net.blobs[layer_name].data[0]
            _ = plt.hist(feat.flatten(), bins=20)
            plt.xlabel('Activation') 
            plt.ylabel('Count')
            plt.title('Activation Histogram of layer '+ layer_name) 
            plt.savefig(outdir + layer_name +'_cvstep_' + str(cv_step) + '_step_' + str(curstep) + '_hist.png',bbox_inches='tight')
            plt.close() # close this figure to prevent memory leak

          vis_square(solver.net.params['conv1'][0].data.transpose(0,2,3,1))
          plt.savefig(outdir + 'cvstep_'+str(cv_step) + '_step_' + str(curstep) + '_conv1_filter.png',bbox_inches='tight')
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
              pass
              #print("init " + tracknm + " statistics")
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
          force_quit = True
          break
        if os.path.isfile(exp_name + '_quit'):
          force_quit = True
          break
      del solver # stop solver since otherwise it's blocking other nets
      cv_step += 1

  if 'prevOutFd' in locals():
    # close the file handles
    os.dup2(prevOutFd, sys.stdout.fileno())
    os.close(prevOutFd)
    os.dup2(prevErrFd, sys.stderr.fileno())
    os.close(prevErrFd)


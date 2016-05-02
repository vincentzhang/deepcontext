import sys
import utils as ut
from multiprocessing import Process, Queue, Pool
import matplotlib.pyplot as plt
import numpy as np
import os, subprocess, time, math, random, shutil, signal, glob, re
#from caffe import layers as L, params as P, to_proto, NetSpec, get_solver, Net
#from caffe.proto import caffe_pb2
#import caffe
import deepcontext_config

exp_name=ut.mfilename()
tmpdir = deepcontext_config.tmp_dir + '/' + exp_name + '_out/';

fnm=tmpdir
np.save(fnm, data)
dataq.put((fnm, perm, label), timeout=600)
fils=glob.glob(outdir + 'model_iter_*.solverstate');

if(len(fils)>0):
  idxs=[];
  for f in fils:
    idxs.append(int(re.findall('\d+',os.path.basename(f))[0]));
  # Load the latest solver state
  idx=np.argmax(np.array(idxs))
  solver.restore(outdir + os.path.basename(fils[idx]));


# code to read the patch and display in correct position
# below is example
# original grid:
# data[1]
# data[0]
# data[2]
p1 = data[1,...]
p2 = data[0,...]
p3 = data[2,...]
con_img =
np.concatenate((p1.transpose(1,2,0),p2.transpose(1,2,0),p3.transpose(1,2,0)))/255
plt.imshow(con_img, cmap=plt.cm.gray)
plt.show()

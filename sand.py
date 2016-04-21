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
import inspect
from mako.template import Template
from mako import exceptions
from mako.lookup import TemplateLookup
import pdb
from scipy import misc
import train as tp

if __name__ == '__main__':
    batch_sz = 100
    patch_sz=(96,96) # size of sampled patches
    exp_name=ut.mfilename();
    outdir = deepcontext_config.out_dir + '/' + exp_name + '_out/';
    tmpdir = deepcontext_config.tmp_dir + '/' + exp_name + '_out/';
    if not os.path.exists(tmpdir):
      os.makedirs(tmpdir)
    if not os.path.exists(outdir):
      os.mkdir(outdir)

    imgs = tp.load_imageset()

    imgsord=np.random.permutation(len(imgs['filename']))
    curidx = 0
    im=ut.get_resized_image(imgsord[curidx % len(imgs['filename'])],
                            imgs,
                            {"gri_targpixels":random.randint(150000,450000)})
    pats = []
    pats=map(tp.prep_image, pats)
    dataq=[]
    procs=[]
    i=0
    tp.imgloader(dataq, batch_sz, imgs, tmpdir,(hash(outdir)+i) % 1000000, i,
              patch_sz)


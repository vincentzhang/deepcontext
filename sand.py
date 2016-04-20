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
import train

if __name__ == '__main__':
    imgs = train.load_imageset()
    imgsord=np.random.permutation(len(imgs['filename']))
    #idx = imgsord[0]
    curidx = 0
    pdb.set_trace()
    #im = ut.get_image(idx,imgs)
    im=ut.get_resized_image(imgsord[curidx % len(imgs['filename'])],
                            imgs,
                            {"gri_targpixels":random.randint(150000,450000)})

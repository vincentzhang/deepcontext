import inspect
import re
from mako.template import Template
from mako import exceptions
from mako.lookup import TemplateLookup
import os
from scipy import misc
import numpy as np
import caffe
import matplotlib.pyplot as plt
import pdb

def mfilename():
  """
    the filename for the root file that's being run.
  """
  stack = inspect.stack();
  filepath = (stack[1][1]);
  dotidx=[m.start() for m in re.finditer('\.', filepath)]
  filenm = filepath[0:dotidx[len(dotidx)-1]];
  dotidx=[m.start() for m in re.finditer('\/', filepath)]
  try:
    # this command fails in python <3
    filenm = filenm[dotidx[len(dotidx)-1]+1:];
  except:
    pass;
  return filenm;

def mkorender(tplfile,outfile,*args,**kwargs):
  """
        create a template from tplfile and call mkorendertpl render it to prototxt
  """
  mylookup = TemplateLookup(directories=[os.getcwd()])
  tpl=Template(filename=tplfile,lookup=mylookup);
  mkorendertpl(tpl,outfile,*args,**kwargs);

def mkorendertpl(tpl,outfile,*args,**kwargs):
  """
        render a template and save it to prototxt
  """
  with open(outfile,"w") as f:
    try:
      f.write(tpl.render(*args,**kwargs));
    except:
      print((exceptions.text_error_template().render()))
      raise;

def mkorenderstr(strtpl,outfile,*args,**kwargs):
  mylookup = TemplateLookup(directories=[os.getcwd()])
  tpl=Template(strtpl,lookup=mylookup);
  mkorendertpl(tpl,outfile,*args,**kwargs);

def get_image(idx,dataset):
  """
    Returns an image, if image is grayscale, replicate to 3 channels
    The range of the data is [0,255]
    Output Dimension: height, width, channel
  """
  im=misc.imread(dataset['dir'] + dataset['filename'][idx])
  if (len(im.shape)==2):
    # if grayscale, stack three channels
    # img shape: height, width, channel, example: 1111x1111x3
    im=np.concatenate((im[:,:,None],im[:,:,None],im[:,:,None]),axis=2)
  return im

def get_resized_image(idx,dataset,conf={}):
  """
        idx: image index
        dataset: imgs{'dir','filename'}
        conf: dictionary 
  """
  targpixels=None;
  if 'targpixels' in dataset:
    targpixels=dataset['targpixels']
  if 'gri_targpixels' in conf:
      # total pixels
    targpixels=conf['gri_targpixels'];
  maxdim=conf.get('gri_maxdim',None);
  try:
    im=get_image(idx,dataset)
  except:
    print "Exception when reading image: ", idx, " ,pixel: ", targpixels
    raise
  if(len(im.shape)==2):
    print("found grayscale image");
    im=np.array([im,im,im]).transpose(1,2,0);
  elif(im.shape[2]==4):
    print("found 4-channel png with channel 4 min "+str(np.min(im[:,:,3])));
    im=im[:,:,0:3];

  # Rescaling the intensity to [0,1], 1: white
  im=im.astype(np.float32)/255;
  if targpixels is not None:
    npixels = float(im.shape[0]*im.shape[1]) # width * height, img is read by
                                            # misc.read
    # TODO: this has issues with the image boundary, as it samples zeros 
    # outside the image bounds.
    # Assume it's square image
    try:
      # this function uses spline interpolation == 1 by default
      im=caffe.io.resize_image(im,(int(im.shape[0]*np.sqrt(targpixels/npixels)),int(im.shape[1]*np.sqrt(targpixels/npixels))))
    except:
      print "Exception when calling io.resize_image"
      raise
  elif maxdim is not None:
    immax=max(im.shape[0:2]);
    ratio=float(maxdim)/float(immax);
    im=caffe.io.resize_image(im,(int(im.shape[0]*ratio),int(im.shape[1]*ratio)))
  return im

def dispdata(net,idx,layer='data'):
  dat=net.blobs[layer].data[idx,:,:,:].transpose((1,2,0));
  dat=(dat-np.min(dat))/(np.max(dat)-np.min(dat));
  plt.imshow(dat);plt.show();



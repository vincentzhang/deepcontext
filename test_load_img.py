import deepcontext_config

def load_imageset():
  datadir=deepcontext_config.imagenet_dir
  imgs={}
  imgs['dir']=datadir+'train/'
  names=[]
  with open(datadir + 'train.txt', 'rb') as f:
    for line in f:
      row=line.split()
      names.append(row[0])
  imgs['filename']=names
  return imgs

img =  load_imageset()
print img['dir']
print img['filename'][0]



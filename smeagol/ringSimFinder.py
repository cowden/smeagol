#!/usr/bin/env python2

###########################################
# Generate a set of single ring images.
# Then, convolve each image with a suitable kernel.
# The data is stored to disk.
############################################


import os,sys,re
import getopt

import pandas as pd
import numpy as np
import scipy.signal as sig
import tables as tb

dir_name = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_name)

import RingGenerator as rg
import RingKernel as rk


storeName = 'dataStore.hf5'
chunk = 250
rRing = 0.5

# parse command line arguments
opts,args = getopt.getopt(sys.argv[1:],'f:c:r:',['store=','chunk-size=','ring-radius='])

for opt,arg in opts:

  if opt in ('-f','--store'):
    storeName = str(arg)

  elif opt in ('-c','--chunk-size'):
    chunk = int(arg)

  elif opt in ('-r','--ring-radius'):
    rRing = float(arg)


# check if the data is there
if os.path.isfile(storeName):
  hdfstore = tb.open_file(storeName,'a')
else:
  sys.stderr.write('ERROR: Cannot open file %s\n' % storeName)
  sys.stderr.flush()
  
  sys.exit(1)


# create kernel
krn = rk.kernel(rRing,0.01)
krn_im = krn.digitize(100,10./500.)


# check for the data and label nodes
nodes = {n.name:n for n in hdfstore.list_nodes('/')}
hasNodes = 'data' in nodes.keys() and 'labels' in nodes.keys()

if not hasNodes:
  sys.stderr.write('Data store does not contain data and label nodes ... bailing out.\n')
  sys.exit(2)



# check if the convolution node already exists
if 'convs' not in nodes.keys():
  print 'Convolution node does not exist ... creating it'
  atom = tb.Atom.from_dtype(np.dtype('float64'))
  hdfstore.create_earray('/','convs',atom=atom,shape=(0,500*500))
  nodes = {n.name:n for n in hdfstore.list_nodes('/')}



# determine the number of chunks
nSamples = nodes['data'].shape[0]
nChunks = nSamples/chunk + 1


# cycle over the  chunks
for c in range(nChunks):
  start = c*chunk
  stop = chunk + start if chunk + start < nSamples else nSamples

  data = nodes['data'][start:stop] 


  # convolve kernel with rings
  convs = []
  for i in range(len(data)):
    convs.append(sig.fftconvolve(data[i].reshape(500,500),krn_im,mode='same'))

    sys.stdout.write('\rConvolving Chunk %d [%d:%d] - %.2f%% Complete' % (c,start,stop,float(i)*100./nSamples)) 
    sys.stdout.flush()


  sys.stdout.write('\n')
  sys.stdout.flush()



  # store the convolutions
  d0 = len(convs)
  print 'Appending  (%d,%d)' % (d0,500*500)
  nodes['convs'].append(np.array(convs).reshape((d0,500*500)))



# store distance and labels

hdfstore.close()




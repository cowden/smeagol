#!/usr/bin/env python2

######################################
# validate the single ring model
# generate a bunch of single ring images, 
# convolve with a kernel and apply maximum_filter
# to select peaks.
# Estimate the accuracy of this algorithm.
######################################


import os,sys
import tables as tb
import getopt

import numpy as np
from scipy.ndimage.filters import maximum_filter
import scipy.signal as sig

dir_name = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_name)

import RingGenerator as rg
import RingKernel as rk

# parse command line arguments
nBatches = 10
nSamples = 100
nHitPerRing = 1000
rRing = 0.5
storeName = 'validation_store_0.hf5'

seed = 42

opts,args = getopt.getopt(sys.argv[1:],'b:n:h:r:f:s:',['batches=','samples=','ring-hits=','ring-radius=','store=','seed='])

for opt,arg in opts:

  if opt in ('-b','--batches'):
    nBatches = int(arg)

  elif opt in ('-n','--samples'):
    nSamples = int(arg)

  elif opt in ('-h','--ring-hits'):
    nHitPerRing = int(arg)

  elif opt in ('-r','--ring-radius'):
    rRing = float(arg)

  elif opt in ('-f','--store'):
    storeName = str(arg)

  elif opt in ('-s','--seed'):
    seed = int(arg)


np.random.seed(seed)

# connect to hdfstore
hdfstore = tb.open_file(storeName,mode='w')
atom = tb.Atom.from_dtype(np.dtype('float64'))
hdfstore.create_earray('/','peak',atom=atom,shape=[0])
hdfstore.create_earray('/','dist',atom=atom,shape=[0])


screen = rg.Screen(domain=[[-5.,5.],[-5.,5.]],N=500)
krn = rk.kernel(rRing,0.01)
krn_im = krn.digitize(100,10./500.)

# cycle over batches
for b in range(nBatches):
  print 'Starting batch %d of %d' % (b,nBatches)
  # generate some rings
  srg = rg.SingleRingGenerator(nHitPerRing,rRing)
  srg.run(nSamples)

  label_inds = np.array([screen.transform_index(np.array([l[1:]]))[0] for l in srg.labels_])

  # apply kernel convolution
  print 'Convolving kernel'
  convs = []
  dists = []
  for i in range(nSamples):
    convs.append(sig.fftconvolve(srg.data_[i].reshape((500,500)),krn_im,mode='same'))

    # apply maximum filter
    mask = (maximum_filter(convs[i],size=25) == convs[i])
    convs[i] *= mask

    # calculate distances
    dists.append(np.array([np.linalg.norm(np.unravel_index(j,(500,500))-label_inds[i]) for j in range(500*500)]))
   

  convs = np.array(convs).flatten()
  dists = np.array(dists).flatten()
  
  # select peaks
  peaks = convs[convs > 150.]
  pdists = dists[convs > 150.]

  print 'Putting data in data store'
  hdfstore.get_node('/','peak').append(convs[convs > 150.].flatten())
  hdfstore.get_node('/','dist').append(dists[convs > 150.].flatten())




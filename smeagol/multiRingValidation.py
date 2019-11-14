

################################
# validate the multiring model
# generate a bunch of multi ring images,
# convolve with a kernel and apply the maximum_filter
# to select peaks.
# Estimate the accuracy of this algorithm.
################################



import os,sys
import tables as tb
import getopt
import pickle as pkl

import numpy as np
from scipy.ndimage.filters import maximum_filter
import scipy.signal as sig

dir_name = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_name)
sys.path.append(os.path.join(os.getenv('HOME'),'proj/CowTools/python'))

from progress import ProgressBar

import RingGenerator as rg
import RingKernel as rk

# parse command line arguments
nBatches = 10
nSamples = 100
nHitPerRing = 1000
rRing = 0.5
nRings = 100
storeName = 'multi_validation_store_0.hf5'

seed = 42


opts,args = getopt.getopt(sys.argv[1:],'b:n:h:r:N:f:s:',['batches=','samples=','ring-hits=','ring-radius=','num-rings=','store=','seed='])

for opt,arg in opts:

  if opt in ('-b','--batches'):
    nBatches = int(arg)

  elif opt in ('-n','--samples'):
    nSamples = int(arg)

  elif opt in ('-h','--ring-hits'):
    nHitPerRing = int(arg)

  elif opt in ('-r','--ring-radius'):
    rRing = float(arg)

  elif opt in ('-N','--num-rings'):
    nRings = int(arg)

  elif opt in ('-f','--store'):
    storeName = str(arg)

  elif opt in ('-s','--seed'):
    seed = int(arg)


np.random.seed(seed)


# connect to hdfstore
#hdfstore = tb.open_file(storeName,mode='w')
#atom = tb.Atom.from_dtype(np.dtype('float64'))
#hdfstore.create_earray('/','peak',atom=atom,shape=[0])
#hdfstore.create_earray('/','dist',atom=atom,shape=[0])


screen = rg.Screen(domain=[[-5.,5.],[-5.,5.]],N=500)
krn = rk.kernel(rRing,0.01)
krn_im = krn.digitize(100,10./500.)

hist = None

# cycle over batches
for b in range(nBatches):

  # generate rings
  mrg = rg.MultiRingGenerator(nHitPerRing,rRing,nRings,sig=3.)
  mrg.run(nSamples)

  # create kernel image and convolve with images
  print 'Convolving kernel to images and calculating distances'
  pb = ProgressBar(nSamples,frequency=1)

  convs = []
  dists = []
  for i in range(len(mrg.data_)):
    convs.append(sig.fftconvolve(mrg.data_[i].reshape((500,500)),krn_im,mode='same'))

    mask = (maximum_filter(convs[i],size=25) == convs[i])
    convs[i] *= mask

    label_inds = screen.transform_index(np.array(mrg.labels_[i]))

    n_labs = len(mrg.labels_[i])

    dists.append(
      np.array([np.min([np.linalg.norm(np.unravel_index(j,(500,500)) - label_inds[k]) for k in range(n_labs)]) for j in range(500*500)])
    )

    pb.update(1)

  pb.complete()
   
  convs = np.array(convs).flatten()
  dists = np.array(dists).flatten()

  data = np.array([d for d in zip(convs[convs > 0],dists[convs > 0])])
  htmp, edges = np.histogramdd(data,bins=[np.linspace(0.,18000.,100),np.linspace(0.,200.,100)])
#
  if hist is None:
    hist = htmp.copy()
  else:
    hist += htmp


 
with open(storeName,'wb') as f:
  pkl.dump(hist,f) 
  

   

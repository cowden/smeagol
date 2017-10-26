

#################################
# C S Cowden
# Classes and methods for ring finder
# algorithm
#################################


import os,sys
import numbers

import numpy as np
import pandas as pd

from scipy.sparse import lil_matrix
from scipy.ndimage.filters import maximum_filter

# import the needed RingFinder tools
import RingKernel as rk
import RingGenerator as rg


###################################################
# Module Classes

class RingFinder(object):

  def __init__(self,Nr=10,span=[0.25,0.75],sigma=0.01
    ,band=0.1,px_width=0.02,max_filter_width=3
    ,thresholds=None
  ):
    '''Initialize the model with all necessary parameters.'''

    # selection thresholds (1 per kernel)
    if thresholds is None:
      raise TypeError('Thresholds cannot be None')

    if isinstance(thresholds,numbers.Number):
      self._thresholds = [thresholds for r in range(Nr)]

    elif hasattr(thresholds,'__iter__'):
      self._thresholds = thresholds
      if len(thresholds) != Nr:
	raise ValueError('Number of thresholds does not match the number of kernels.')

    else:
      raise TypeError('Don\'t know how to deal with type of thresholds %s' % (str(type(thresholds))))

    # collect hyperparameters
    self._sigma = sigma
    self._band = band
    self._px_width = px_width
    self._max_filter_width = max_filter_width
    self._Nr = Nr
    self._span = span

    # list of radii
    self._deltaR = (span[1] - span[0])/(Nr-1)
    self._radii = list(np.arange(span[0],span[1]+0.5*self._deltaR,self._deltaR))
  
    # create the kernel convolvers
    kernel_params = {
      's': self._sigma
      ,'F': self._max_filter_width
      ,'krn_pixel_width': self._px_width
      ,'krn_N': int(2.7*span[1]/self._px_width)
    }
    self._kernels = [rk.KernelConvolver(r,**kernel_params) for r in self._radii]

  
  def eval(self,images):
    '''Evaluate the model on a set of image.'''

    # check the dimensions of the image
    # call overlay to overlay and evaluate the kernel convolvers
    data = self._overlay(images)

    # remove overlaps from nearby radii
    data = self._removeOverlaps(data)

    # collect labels
    labels = self._collectLabels(data)

    return labels


  def _overlay(self,images):
    '''Execute the kernel convolvers on the images and overlay the resulting images.'''

    # execute kernels 
    for i in range(self._Nr):
      self._kernels[i].batch(images)

    # create the dataset
    # a list of sparse matrices (one for each image)
    # an image is represented as a sparse matrix [Nr,nPx,nPx]
    n_events = images.shape[0]
    data = [ np.array([k.convs_[i].todense() for k in self._kernels]) for i in range(n_events) ]

    return data    


  def _removeOverlaps(self,data):
    '''Remove overlap from overlay data.'''

    # determine the size of the footprint in the radius direction
    r_footprint = int(self._band/2./self._deltaR)
    size = (r_footprint,self._max_filter_width,self._max_filter_width)

    # let's try this with a maximum_filter
    # unpack sparse matrix, apply maximum_filter, and resparsify
    data = [d*(maximum_filter(d,size=size)==d) for d in data]

    return data


  def _collectLabels(self,data):
    '''Collect labels from input data.'''


    # collect labels for each image and radius
    labels = [ [(self._radii[r],np.argwhere(d[r] > self._thresholds[r])) for r in range(self._Nr)] for d in data ]
   

    return labels

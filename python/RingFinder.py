

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

from sklearn.linear_model import LogisticRegression

# import the needed RingFinder tools
import RingKernel as rk
import RingGenerator as rg
import Metrics as met


###################################################
# Module Classes

class RingFinder(object):
  '''
    This class implements a regression to separate and identify rings.
  '''

  def __init__(self,screen,Nr=10,span=[0.25,0.75],sigma=0.01,bk_sample=100,window=[50,50]
  ):
    '''Initialize the model with all necessary parameters.'''

    # collect hyperparameters
    self._screen = screen
    self._sigma = sigma
    self._Nr = Nr
    self._span = span
    self._n_sample = bk_sample
    self._window = window

    # list of radii
    self._deltaR = (span[1] - span[0])/(Nr-1)
    self._radii = list(np.arange(span[0],span[1]+0.5*self._deltaR,self._deltaR))
 


  def _randomlocation(self,data):
    '''
      Select a random location for a window.
      Return a random image and location in that image.
    '''

    # pick a random image
    n = data.shape[0]
    indx = np.random.choice(n)

    # choose a window
    pos = np.random.random_integers(0,data.shape[1],size=2)

    return indx,pos



  def _getkernel(self,r,s,w,shape):
    '''
      Get a kernel window.  Use this to manually overllay a window
      to calculate distributional moments.
  
    '''
 
    window = rk.kernel(r,s).digitize(shape[0],w)
  
    return window



  def _getwindow(self,image,index,shape):
    '''
      Given a particular image and set of indices, return the 
      window of the specified shape around the indices from 
      the input image.
    '''
  
    window = np.zeros(shape)
  
    x_range = np.array([index[0] - shape[0]/2, index[0] + shape[0]/2],dtype=int)
    y_range = np.array([index[1] - shape[0]/2, index[1] + shape[0]/2],dtype=int)
  
    windx = [[0,51],[0,51]]
  
    if x_range[0] < 0: 
      windx[0][0] = np.abs(x_range[0])
      x_range[0] = 0
   
    if y_range[0] < 0:
      windx[1][0] = np.abs(y_range[0])
      y_range[0] = 0
  
    if x_range[1] > image.shape[1]:
      windx[0][1] = shape[0] - x_range[1] + image.shape[0] 
  
    if y_range[1] > image.shape[1]:
      windx[1][1] = shape[0] - y_range[1] + image.shape[0]
  
    window[windx[0][0]:windx[0][1],windx[1][0]:windx[1][1]] = image[x_range[0]:x_range[1],y_range[0]:y_range[1]]
  
    return window




  def _getcoordinates(self,image,w=0.02):
    '''
      return and array of coordinates corresponding to the 
      flattened image array.
    '''
 
    return np.array(np.unravel_index(np.arange(np.multiply(*image.shape)),image.shape)).T*w - image.shape[0]*w/2



  def _transformwindow(self,X):
    '''
      transfrom the X space into polar coordinates.
    '''
  
  
    R = np.zeros(X.shape)
  
    R[:,0] = np.sqrt(X[:,0]**2 + X[:,1]**2)
    R[:,1] = np.arctan2(X[:,0],X[:,1]) + np.pi
  
    return R



  def _getSignalAndVariance(self,image,index,radius,s=0.01,w=0.02,shape=[50,50]):
    '''
     Return the convolution strength and polar variance in the window of shape
     around the index in the image.  Set the kernel radius to `radius`.
    '''
 
    # get the kernel
    kern = self._getkernel(radius,s,w=w,shape=shape)
  
    # get the window
    window = self._getwindow(image,index,shape=shape)
  
    # transform the indices
    R = self._transformwindow(self._getcoordinates(window,w))
  
    # compute the convolution
    density = window*kern
    conv = np.sum(density)
    density /= conv
  
    # compute the variance
    mu = np.sum(R[:,1]*density.flatten())
    tvar = np.sum((R[:,1]-mu)**2*density.flatten())
  
    if np.isnan(tvar):
      tvar = 0.
  
    return conv,tvar



  def _calcConv(self,images,rdf):
    '''
      From the provided set of images, calculate the convolution
      strengths at the points and radii specified in the dataframe.
    '''

    return rdf[['image','x','y','r']].apply(lambda x: self._getSignalAndVariance(images[int(x[0])],[int(x[1]),int(x[2])],x[3],s=self._sigma,w=self._screen._pixel_width,shape=self._window)[0],axis=1)



  def _calcTvar(self,images,rdf):
    '''
      From the provided set of images, calculate the variance in kernel theta
      strengths at the points and radii specified in the dataframe.
    '''

    return rdf[['image','x','y','r']].apply(lambda x: self._getSignalAndVariance(images[int(x[0])],[int(x[1]),int(x[2])],x[3],s=self._sigma,w=self._screen._pixel_width,shape=self._window)[1],axis=1)


  def _ihs(self,x,lam=1):
    '''Inverse hyperbolic sine'''

    return np.log(x/lam + np.sqrt((x/lam)**2+1))
 
  def eval(self,images):
    '''Evaluate the model on a set of image.'''



  def train(self,images,labels):
    '''
    '''

    N = images.shape[0]

    # find the actuals (centers)
    actuals = met.prepare_actuals(labels,self._screen)

    # generate some random locations and radii
    rand = [self._randomlocation(images) for i in range(100*self._n_sample*N)]
    rand = [[r[0],r[1][0],r[1][1],np.random.uniform(*self._span)] for r in rand]

    # prepare a ring dataframe of actuals and random windows/radii 
    self._rdf = pd.concat(
      [pd.DataFrame(
        {'is_ring':[1]*len(actuals[i])
        ,'image':[i]*len(actuals[i])
        ,'x':[a[0] for a in actuals[i]]
        ,'y':[a[1] for a in actuals[i]]
        ,'r':[a[2] for a in actuals[i]]
        }) for i in range(len(actuals))] +
      [pd.DataFrame(
        {'is_ring':[0]*len(rand)
        ,'image':[r[0] for r in rand]
        ,'x':[r[1] for r in rand]
        ,'y':[r[2] for r in rand]
        ,'r':[r[3] for r in rand]
        })]
    )

    self._rdf = self._rdf.sample(frac=1).reset_index(drop=True) 

    # compute the covariance
    self._rdf['conv'] = self._calcConv(images,self._rdf)
  
    # compute the variance in theta
    self._rdf['tvar'] = self._calcTvar(images,self._rdf)

    # transform the data
    tvar_lambda = np.mean(self._rdf['tvar'])/(2*np.sqrt(3))
    self._rdf['tvarT'] = self._ihs(self._rdf['tvar'],tvar_lambda)

    conv_lambda = np.mean(self._rdf['conv'])/(2*np.sqrt(3))
    self._rdf['convT'] = self._ihs(self._rdf['conv'],conv_lambda)

    # fit the data
    X = np.zeros((len(self._rdf),6))
    X[:,0] = self._rdf['convT']
    X[:,1] = self._rdf['tvarT']
    X[:,2] = self._rdf['r']
    X[:,3] = self._rdf['convT']*self._rdf['tvarT']
    X[:,4] = self._rdf['convT']*self._rdf['r']
    X[:,5] = self._rdf['tvarT']*self._rdf['r']

    y = self._rdf['is_ring']

    self._clf = LogisticRegression(solver='lbfgs',penalty='none')
    self._clf.fit(X,y)


    # prepare performance statistics
    #------------------------------
    # compute the correlation matrix

    # compute coefficient correlation matrix

    # compute the z-scores and test of coefficients

    # compute the R^2

    # compute the likelihood

    # 



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
from scipy.ndimage import filters
import scipy.signal as sig

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
    pos = np.random.random_integers(0,data.shape[1]-1,size=2)

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

  def _getConv(self,image,index,radius,s=0.01,w=0.02,shape=[50,50]):
    '''
      Return the convolution strength at a given point.
    '''

    # get the kernel
    kern = self._getkernel(radius,s,w=w,shape=shape)
    kern /= np.sum(kern)

    # compute the convolution
    density = sig.fftconvolve(image,kern,mode='same')

    return density[index[0],index[1]]

  def _getConvVariance(self,image,index,radius,s=0.01,w=0.02,shape=[50,50]):
    '''
      Return the  polar and variance at a given point.
    '''

    # get the kernel
    kern = self._getkernel(radius,s,w=w,shape=shape)
    kern /= np.sum(kern)

    # get the polar window coordinates
    Xdig = np.array(np.unravel_index(np.arange(np.prod(shape)),shape)).T*w - w*shape[0]/2
    R = np.arctan2(Xdig[:,0],Xdig[:,1]) + np.pi
    R = R.reshape(shape)

    # compute the convolutions
    norm_factor = sig.fftconvolve(image,kern,mode='same')
    conv1 = sig.fftconvolve(image,R**2*kern,mode='same')/norm_factor
    w = sig.fftconvolve(image,R*kern,mode='same')/norm_factor

    return (conv1 - w**2)[index[0],index[1]]


  def _calcConv(self,images,rdf):
    '''
      From the provided set of images, calculate the convolution
      strengths at the points and radii specified in the dataframe.
    '''

    return rdf[['image','x','y','r']].apply(lambda x: self._getConv(images[int(x[0])],[int(x[1]),int(x[2])],x[3],s=self._sigma,w=self._screen._pixel_width,shape=self._window),axis=1)



  def _calcTvar(self,images,rdf):
    '''
      From the provided set of images, calculate the variance in kernel theta
      strengths at the points and radii specified in the dataframe.
    '''

    return rdf[['image','x','y','r']].apply(lambda x: self._getConvVariance(images[int(x[0])],[int(x[1]),int(x[2])],x[3],s=self._sigma,w=self._screen._pixel_width,shape=self._window),axis=1)


  def _ihs(self,x,lam=1):
    '''Inverse hyperbolic sine'''

    return np.log(x/lam + np.sqrt((x/lam)**2+1))


  def _score(self,image):
    '''
      Score a single image.
    '''

    # prepare the kernels
    kernels = [rk.kernel(r,self._sigma).digitize(self._window[0],self._screen._pixel_width) for r in self._radii]
    kernels = [kr/np.sum(kr) for kr in kernels] 

    # digitize the window centers
    Xdig = np.array(np.unravel_index(np.arange(np.prod(self._window)),self._window)).T*self._screen._pixel_width - self._screen._pixel_width*self._window[0]/2
    R = np.arctan2(Xdig[:,0],Xdig[:,1]) + np.pi
    R = R.reshape(self._window)

    zeta2 = np.zeros((len(self._radii),*image.shape))
    convs = np.zeros((len(self._radii),*image.shape))

    for i,r in enumerate(self._radii):

      # execute the convolution
      convs[i,:,:] = sig.fftconvolve(image,kernels[i],mode='same')
  
      tmpconv = sig.fftconvolve(image,R**2*kernels[i],mode='same')/convs[i,:,:]
      w = sig.fftconvolve(image,R*kernels[i],mode='same')/convs[i,:,:]

      # calculate the theta variance
      zeta2[i,:,:] = (tmpconv - w**2)

    # prepare the features
    X = np.zeros((image.size*len(self._radii),7))

    # set the pixel index
    X[:,6] = np.array([np.arange(image.size)]*len(self._radii)).flatten()

    # set the radius
    X[:,2] = np.array([[r]*image.size for r in self._radii]).flatten()

    # set the convolution
    X[:,0] = self._ihs(convs.flatten(),self._conv_lambda)

    # set the zeta2
    X[:,1] = self._ihs(zeta2.flatten(),self._tvar_lambda)

    # compute interaction terms
    X[:,3] = X[:,0]*X[:,1]
    X[:,4] = X[:,0]*X[:,2]
    X[:,5] = X[:,1]*X[:,2]

    # score the model
    preds = self._clf.predict_proba(X[:,:6])[:,1]
    preds = np.log(preds/(1-preds))

    # reshape and return
    return preds.reshape((len(self._radii),*image.shape)) 


  #
  def _filter_image(self,image,size):
    '''
      Apply the maximum filter to the image with the given window size.

      Parameters
      ---------
      image (numpy.array) - the input image, usually scores.
      size (int or tuple) - the footprint/shape/window for the filter.

      Returns
      -------
      the filtered image
    '''

    im = np.zeros(image.shape)
    index = image == filters.maximum_filter(image,size)
    im[index] = image[index]

    return im


  def _selectByFilter(self,scores,size=(2,2)):
    '''
      Apply _filter_image to each radius in the scan.
    '''

    fscores = np.zeros(scores.shape)
    for i in range(scores.shape[0]):
      fscores[i] = self._filter_image(scores[i],size)

    return fscores


  def _localize(self,scores):
    '''
      localize the rings.
    '''

    scores = self._selectByFilter(scores,size=(2,2))
    scores = self._selectByFilter(scores,size=(10,10))
    indx = scores == filters.maximum_filter(scores,(3,10,10))
    z = -np.inf*np.ones(scores.shape)
    z[indx] = scores[indx]
    scores = z

    return scores



  # -------
  # "public" methods
 
  def eval(self,images):
    '''Evaluate the model on a set of image.'''

    N = images.shape[0]
   
    # iterate over the images 
    for i in range(N):

      # score the image
      scores = self._score(images[i])

      # localize the rings
      scores = self._localize(scores)

      #


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
    self._rdf['conv'] = self._calcConv(images,self._rdf,)
  
    # compute the variance in theta
    self._rdf['tvar'] = self._calcTvar(images,self._rdf)

    # transform the data
    self._tvar_lambda = np.mean(self._rdf['tvar'])/(2*np.sqrt(3))
    self._rdf['tvarT'] = self._ihs(self._rdf['tvar'],self._tvar_lambda)

    self._conv_lambda = np.mean(self._rdf['conv'])/(2*np.sqrt(3))
    self._rdf['convT'] = self._ihs(self._rdf['conv'],self._conv_lambda)

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

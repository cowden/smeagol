


############################################
#
############################################


import numpy as np
from scipy.sparse import lil_matrix
import scipy.signal as sig
from scipy.ndimage.filters import maximum_filter

import RingGenerator as rg



class kernel(object):

  def __init__(self,R,s):
    '''Define a kernel functing having radius R and gaussian width s.'''

    self.R_ = R
    self.sigma_ = s

    self.inv_norm_ = np.sqrt(2.)*np.pi*s*(2.*np.sqrt(np.pi)*R+np.sqrt(2)*s*np.exp(-R**2/(2.*s**2)))
    self.norm_ = 1./self.inv_norm_


  def eval(self,points):
    '''Evaluate the kernel function at the specified points.'''

    return self.norm_*np.exp(-(np.sqrt(np.array(points)[:,0]**2+np.array(points)[:,1]**2)-self.R_)**2/(2.*self.sigma_**2))

  def cov(self,points):
    '''Evaluate the covariance of R and theta from the specified points, weighted by the kernel itself.'''

    # get the weights
    w = self.eval(points)

    # transform the points
    polar = np.zeros(points.shape)
    polar[:,0] = np.sqrt(points[:,0]**2+points[:,1]**2)
    polar[:,1] = np.arctan2(points[:,1],points[:,0])

    # return the covariance
    return np.cov(polar.T,aweights=w)


  def peval(self,x,y):
    '''Evaluate the function at a single point.'''

    return self.eval([[x,y]])[0]


  def digitize(self,N,pixel_width):
    '''Return a digitized kernel image.  Assume a square image
     with N pixels on a side where each pixel is of the given 
     width.  One should be careful to align this pixel width to that
    of the target images.'''

    dd = pixel_width*N/2.
    krn_scrn = rg.Screen(domain=[[-dd,dd],[-dd,dd]],N=N)

    return self.eval(list(zip(*[x.flatten() for x in krn_scrn.centers()]))).reshape(N,N)


  def cov_digitize(self,N,pixel_width):
    dd = pixel_width*N/2.
    krn_scrn = rg.Screen(domain=[[-dd,dd],[-dd,dd]],N=N)

    x = np.array(list(zip(*[x.flatten() for x in krn_scrn.centers()])))
    w = self.eval(x)
    t = np.arctan2(x[:,1],x[:,0])
    
    return ((t*w)**2).reshape(N,N)

  def mean_digitize(self,N,pixel_width):
    
    dd = pixel_width*N/2.
    krn_scrn = rg.Screen(domain=[[-dd,dd],[-dd,dd]],N=N)

    x = np.array(list(zip(*[x.flatten() for x in krn_scrn.centers()])))
    w = self.eval(x)
    t = np.arctan2(x[:,1],x[:,0])
    
    return ((t*w)).reshape(N,N)


#
# class to
class KernelConvolver(object):
  '''This class encapsulates a ring kernel and provides methods to convolve it with some data.'''

  def __init__(self,R,s,F,**kwargs):
    '''Pass the kernel ring radius and width as well as the size of the maximum filter.'''
  
    self.R_ = R
    self.s_ = s
    self.F_ = F
 
    self.kernel_ = None
    self.krn_im_ = None
    self.krn_N_ = 100
    self.krn_pixel_width_ = 10./500.

    self.do_maxFilter_ = True

    self.screen_ = None
    self.domain_ = [[-5.,5.],[-5.,5.]]
    self.pixels_ = 500


    # parse kwargs
    for k,v in kwargs.items():
      if k == 'kernel':
        self.kernel_ = v
      elif k == 'max_filter':
        self.do_maxFilter_ = bool(v)
      elif k == 'screen':
        self.screen_ = v
      elif k == 'krn_im':
        self.krn_im_ = v
      elif k == 'krn_N':
        self.krn_N_ = int(v)
      elif k == 'krn_pixel_width':
        self.krn_pixel_width_ = float(v)
      elif k == 'screen_pixels':
        self.pixels_ = int(v)
      elif k == 'domain':
        self.domain_ = v

    if self.kernel_ is None:
      self.kernel_ = kernel(self.R_,self.s_)

    if self.krn_im_ is None:
      self.krn_im_ = self.kernel_.digitize(self.krn_N_,self.krn_pixel_width_)

    if self.screen_ is None:
      self.screen_ = rg.Screen(self.domain_,self.pixels_)

    self.convs_ = []
    

  def convolve(self,image):
    '''Perform the convolution.'''
    return sig.fftconvolve(image,self.krn_im_,mode='same')

  def convolution_variance(self,image):
    '''Compute the variance of data in the kernel window.'''

    # transform the x-space

    # compute convolutions

    # compute variance

  def max_filter(self,image):
    '''Perform the maximum filter.'''
    return (maximum_filter(image,size=self.F_) == image)

  def filter(self,image):
    '''Filter the image to select peaks.'''

    self.convs_.append(self.convolve(image))
    if self.do_maxFilter_:
      self.convs_[-1] *= self.max_filter(self.convs_[-1])


    # sparsify the convolution matrix.
    self.convs_[-1] = lil_matrix(self.convs_[-1])    


  def batch(self,batch):
    '''Filter a batch of images.'''

    for image in batch:
      self.filter(image)



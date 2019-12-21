
import numpy as np


#
# get a window from an image
def getwindow(image,index,shape=[50,50]):
  '''
    Given a particular image and set of indices, return the 
    window of the specified shape around the indices from 
    the input image.
  '''

  window = np.zeros(shape)

  x_range = np.array([index[0] - shape[0]/2, index[0] + shape[0]/2],dtype=int)
  y_range = np.array([index[1] - shape[0]/2, index[1] + shape[0]/2],dtype=int)

  windx = [[0,shape[0]+1],[0,shape[1]+1]]

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


#
# get a span for a window
def getwindowspan(index,shape,size):
  
  low = index - shape/2
  high = index + shape/2 + 1
  if low < 0:
    low = 0
  if high >= size:
    high = size

  return int(low),int(high)

def selectWindowSlice(index,shape,screen):
  
  Np = screen.N_
  
  s1 = getwindowspan(index[0],shape[0],Np)
  s2 = getwindowspan(index[1],shape[1],Np)

  return tuple([np.s_[:],np.s_[s1[0]:s1[1]],np.s_[s2[0]:s2[1]]])


  

#
#  return the predicted ring centers
def getPredictedRingCenters(scores,screen,radii,threshold=0.):

  cents = np.argwhere(scores > threshold)
  rs = radii[cents[:,0]]
  cents = cents.astype(float)
  cents[:,:2] = cents[:,1:]
  cents[:,2] = rs

  return cents






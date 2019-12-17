

#########################################
# C S Cowden		22 April 2017
#########################################

import numpy as np
import random
import os,sys,re

import matplotlib as mpl
import pylab as plt

import RingGenerator as rg


#
# A method to plot an array of images
def gridPlot(data,**kwargs):
  '''plot an array of images randomly selected from the input data.
    data is a list of flattened image arrays.  
    array: (nx,xy) the dimensions of the plot array
    dim: (x,y) the dimensions of the image pixels by pixels
  '''

  array = (3,3)
  dim = (500,500)

  # fig size
  fs = (8,8) 

  # label data
  labels = None

  # screen to translate labels
  screen = rg.Screen([[-5.,5.],[-5.,5.]],500)

  # parse kwargs
  for k,v in kwargs.iteritems():
    if k == 'array':
      array = tuple(v)
    elif k == 'dim':
      dim = tuple(v)
    elif k == 'labels':
      labels = np.array(v)
    elif k == 'screen':
      screen = v

  # select the images
  index = random.sample(np.arange(len(data)),array[0]*array[1])

  # create the plot
  fig = plt.figure(figsize=fs)

  for i in range(array[0]):
    for j in range(array[1]):
      ind = i*array[0]+j
      indx = index[ind]
      print('index:',i,j,ind)
      ax = plt.subplot(array[0],array[1],ind+1)
      ax.imshow(data[indx].reshape((dim[0],dim[1])),interpolation='none',extent=[0.,dim[0],0.,dim[1]],origin='lower')

      if labels is not None:
      	pos = screen.transform_index(np.array([labels[indx][1:]]))
      	ax.scatter(pos[:,1],pos[:,0])
      	print(pos[0],np.unravel_index(np.argmax(data[indx]),(dim[0],dim[1])),np.linalg.norm(np.array(pos[0])-np.unravel_index(np.argmax(data[indx]),(dim[0],dim[1]))))

      ax.set_xticklabels([])
      ax.set_yticklabels([])

      ax.set_title('%d' % index[ind])

    


#
# plot a set of rings
def plotRings(data,px_width=0.02):
  '''Given a set of triplets representing the ring center (x,y) and the 
radius, plot each ring.
  '''

  # get the current axis
  ax = plt.gca()

  # draw each circle independently
  for ring in data: 
    circle = plt.Circle((ring[1],ring[0]),ring[2]/px_width,color='r',fill=False)
    ax.add_artist(circle)


#
# plot score in in window along axis
def plotScoreInWindow(data,axis,mplax):
  '''
    Given an array of scores, plot it for each element along a particular axis.
    For example, suppose data is a 10x5x5 array and axis=0, this plots the 25 
   elements over the 10 along axis 0.
  ''' 


  X = np.moveaxis(data,axis,0)
  shape = X.shape
  X = X.reshape((shape[0],np.prod(shape[1:])))
  
  shape = X.shape
  l = np.arange(shape[0])
  for i in range(shape[1]):
    x = X[:,i]

    # plot the curve
    mplax.plot(l,x,color='#333333',ls=':',lw=0.5,alpha=0.5)

  return mplax



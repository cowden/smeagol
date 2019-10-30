

###########################################
# C S Cowden            12 April 2017
###########################################



import numpy as np

import pickle as pkl
import sys


#from progress import ProgressBar

class Screen(object):
  '''Discritization of the imaging layer, e.g. where the light hits.'''

  def __init__(self,domain,N):
    '''Provide the domain [[xmin,xmax],[ymin,ymax]], and number of pixels on a side.'''

    self.domain_ = np.array(domain).copy()
    self.width_ = self.domain_[0,1] - self.domain_[0,0]
    self.height_ = self.domain_[1,1] - self.domain_[1,0]
    self.N_ = N
    self._pixel_width = self.width_/self.N_

    # internal data
    self.data_ = np.zeros((N,N))
   
    # bin boundaries
    self.step_ = 2./(N)
    self.bins_ = np.array([b for b in np.arange(-1.,1.,self.step_)])


  def transform(self,points):
    '''Transform a set of points to scaled coordinate system.'''

    # scaled points
    sp = points.copy()

    translate = np.array([self.domain_[0,0]+self.width_/2., self.domain_[1,0]+self.height_/2.])
    sp -= translate
    sp[:,0] = 2.*sp[:,0]/self.width_
    sp[:,1] = 2.*sp[:,1]/self.height_

    return sp

  def inv_transform(self,spoints):
    '''Inverse transform of the scaled and translated spoints into the true space points.'''

  
    points = spoints.copy()
    points[:,0] = self.width_*points[:,0]/2.
    points[:,1] = self.height_*points[:,1]/2.
  
    points += np.array([self.domain_[0,0]+self.width_/2., self.domain_[1,0]+self.height_/2.])

    return points


  def index_transform(self,indices):
    '''Transform the list of matrix indices to x,y space points.'''

    # assume the data is the form of self.data_
    cents = self.centers()
    
    return np.array([cents[i[0],i[1]] for i in indices]) 


  def discretize(self,points):
    '''discretize the set of points in (x,y) coordinate space.  '''

    ind = self.transform_index(points)
    for i in ind:
      self.data_[i[0],i[1]] += 1
 
    return self


  def transform_index(self,points):
    '''Transform the points to indices into the data.'''

    tp = self.transform(points)
    return np.array(list(zip(np.digitize(tp[:,0],self.bins_),np.digitize(tp[:,1],self.bins_)))) - 1



  def add(self,screen):
    '''Add data to screen image.'''

    self.data_ += screen.data_

    return self



  def clear(self):
    '''Clear the screen/data.'''

    self.data_ = np.zeros((self.N_,self.N_))


  def centers(self):
    '''Return the centers for each bin.'''

    x_step = self.width_/self.N_
    y_step = self.height_/self.N_

    x_centers = np.arange(self.domain_[0,0]+x_step/2.,self.domain_[0,1],x_step) 
    y_centers = np.arange(self.domain_[1,0]+y_step/2.,self.domain_[1,1],y_step)

    xx, yy = np.meshgrid(x_centers,y_centers)

    return (xx,yy)

#
# NoiseGenerator
# Generate random noise
class NoiseGenerator(object):
  '''Generate random, diffuse noise over the surface of the image.'''

  def __init__(self):
    pass

  def generateNoise(self,N,domain):
    '''Cover the domain 2x2 array with uniformly distribution points.  
    The number of points is drawn from a Poisson distribution with mean N.'''

    n = 0
    while n == 0:
      n = np.random.poisson(N)

    w = domain[0][1] - domain[0][0]
    h = domain[1][1] - domain[1][0]
    x = np.random.random_sample(n)*w+domain[0][0]
    y = np.random.random_sample(n)*h+domain[1][0]
    points = np.array(
      list(zip(
        x
        ,y
      ))
    )

    return points[(points[:,0]>domain[0][0]) 
      & (points[:,0]<domain[0][1] )
      & (points[:,1]>domain[1][0] )
      & (points[:,1]<domain[1][1] )
    ]

#
# RingGenerator
# Base class to handle generic routines of ring simulation.
class RingGenerator(object):
  '''Generate a ring for a given set of parameters.'''

  def __init__(self):
    '''Generator tools for MC ring simulation.   '''
    pass


  def generateRing(self,radius,pos,N):
    '''Generate points at a radius around the center (pos), where
    the number of points to generate is a Poisson random number of 
    mean N.'''

    # generate number of points
    n = 0
    while n == 0:
      n = np.random.poisson(N)

    thetas = 2*np.pi*np.random.random_sample(n)
    points = np.array([(radius*np.cos(t) + pos[0],radius*np.sin(t) + pos[1]) for t in thetas]) 

    return points



  def imageRing(self,points,screen):
    '''Apply screen transformations and return discretized result.'''

    # transform points to screen dimensions
    # then discretize data (accumulate) into image
    # return the new screen
    return screen.discretize(points)


  #-----------------------------------
  # virtual methods to be over written by sub-classes

  def run(self):
    '''virtual method to run the generator.  Sub-classes should 
    define this method to perform the specific task for which is
    was built.'''

    raise Exception('This is a purely virtual method of RingGenerator...')  

  def clear(self):
    '''virtual method to clear held data in the class.'''

    raise Exception('This is a purely virtual method of RingGenerator...')



#
# SingleRingGenerator
# Sub-class of RingGenerator to generate a single ring/event.
class SingleRingGenerator(RingGenerator):
  '''Generate a single ring ... inherits from 
  RingGenerator.'''

  def __init__(self,N,R,**kwargs):
    '''Define the event generation paramters.
      N - Number of points per ring (mean of Poisson dist.)
      R - Radius of ring.
      domain - the domain of the image (xy space)
      pixels - number of pixels on one side of image (assume square)
      noise - noise level in image.'''

    self.N_ = N
    self.R_ = R

    self.domain_ = [[-5.,5.],[-5.,5.]]
    self.pixels_ = 500
    self.noise_ = 10000

    for k,v in kwargs.items():
      if k == 'domain':
        self.domain_ = v
      elif k == 'pixels':
        self.pixels_ = int(v)
      elif k == 'noise':
        self.noise_ = v

    # data is an empty list.
    # it will be populated with flattened images
    self.data_ = []
    self.labels_ = []


  def run(self,nEvents):
    '''Simulate nEvents events by parameters already established.'''

    sys.stdout.write('Beginning event generation of %d events.\n' % nEvents)
   
    pos = (0,0) 
    w = self.domain_[0][1] - self.domain_[0][0]
    h = self.domain_[1][1] - self.domain_[1][0]
  
    # main event loop
    for e in range(nEvents):
      
      # create blank screen
      screen =  Screen(self.domain_,self.pixels_)

      # generate the position (x,y)
      pos = (
        w*np.random.random_sample() + self.domain_[0][0] 
        ,h*np.random.random_sample() + self.domain_[1][0]
      )

      # generate the ring
      points = self.generateRing(self.R_,pos,self.N_)


      # clean points
      points = points[(points[:,0]>self.domain_[0][0])
        & (points[:,0]<self.domain_[0][1])
        & (points[:,1]>self.domain_[1][0])
        & (points[:,1]<self.domain_[1][1])
      ]

      # generate noise
      if self.noise_ is not None:
        npoints = NoiseGenerator().generateNoise(self.noise_,self.domain_)
        points = np.array(list(points) + list(npoints))

      # discretize the image
      screen = self.imageRing(points,screen)

      # store image
      self.data_.append(screen.data_.flatten())
      self.labels_.append([self.R_,pos[0],pos[1]])

      sys.stdout.write('\rComplete: %.2f%%' % (float(e)*100./nEvents))
      sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()


  def dumpData(self,outfile):
    '''Store the data in a pickle file.'''

    with open(outfile,'rb') as f:
      pkl.dump({'images':self.data_,'labels':self.labels_},f)



#
# Generate multiple rings per image.
class MultiRingGenerator(RingGenerator):
  '''Generate multi ring images
  Inherit from RingGenerator
  '''

  def __init__(self,N,R,n_rings,**kwargs):
    '''Define the multi-ring event generation.
      N - Number of points per ring (mean of Poisson dist.)
      R - Radius of rings
      n_rings - Number of rings rings/image (mean of a Poisson dist.)
      domain - the domain of the image (xy space)
      noise - noise level in image.
      pixels - number of pixels on one side of image (assume square)
      dist - Distribution from which to draw ring centers
        'uniform' or 'Guassian'
      pos - center of Gaussian 
      sig - symetric width of Guassian 
    '''

    self.N_ = N
    self.R_ = R
    self.nRings_ = n_rings

    self.domain_ = [[-5.,5.],[-5.,5.]]
    self.pixels_ = 500
    self.noise_ = 10000

    self.dist_ = 'gaussian'
    self.pos_ = (0.,0.)
    self.sig_ = 50.

    self.possible_dists_ = set(['gaussian','uniform'])

    for k,v in kwargs.items():
      if k == 'domain':
        self.domain_ = v
      elif k == 'pixels':
        self.pixels_ = int(v)
      elif k == 'noise':
        self.noise_ = v
      elif k == 'dist':
        val = str(v).lower()
        if val not in self.possible_dists_:
          raise ValueError('Not a valid choice of distribution.')
        self.dist_ = val
      elif k == 'pos':
        self.pos_ = tuple(v)
      elif k == 'sig':
        self.sig_ = float(v)

    # data is an empty list.
    # it will be populated with flattened images
    self.data_ = []
    self.labels_ = []


  def gaussian_rings(self):
    '''Generate a set of gaussian ring centers.'''

    n_rings = 0
    while n_rings == 0:
      n_rings = np.random.poisson(self.nRings_)

    return np.random.multivariate_normal(self.pos_,np.identity(2)*self.sig_,n_rings)
 

  def uniform_rings(self):
    n_rings = 0
    while n_rings == 0:
      n_rings = np.random.poisson(self.nRings_)

    w = self.domain_[0][1] - self.domain_[0][0]
    h = self.domain_[1][1] - self.domain_[1][0]

    x = w*np.random.random_sample(n_rings) + self.domain_[0][0]
    y = h*np.random.random_sample(n_rings) + self.domain_[1][0]

    return np.array([[x[i],y[i]] for i in range(n_rings)])
    


  def run(self,nEvents):
    '''Simulate nEvents by parameters already established.'''

    sys.stdout.write('Beginning event generation of %d events.\n' % nEvents)
    #pb = ProgressBar(nEvents)
    

    if self.dist_ == 'gaussian':
      self.labels_ = [self.gaussian_rings() for i in range(nEvents)]
    else:
      self.labels_ = [self.uniform_rings() for i in range(nEvents)]


    for i in range(nEvents):
       
      # blank screen
      screen = Screen(self.domain_,self.pixels_)

      for r in range(len(self.labels_[i])):
        points = self.generateRing(self.R_,self.labels_[i][r],self.N_)
     
        points = points[(points[:,0]>self.domain_[0][0])
          & (points[:,0]<self.domain_[0][1])
          & (points[:,1]>self.domain_[1][0])
          & (points[:,1]<self.domain_[1][1])
        ] 
        
        screen.discretize(points) 

      # generate noise
      if self.noise_ is not None:
        npoints = NoiseGenerator().generateNoise(self.noise_,self.domain_)
        screen.discretize(npoints)

      # store image
      self.data_.append(screen.data_.flatten())

      #pb.update(1)

    #pb.complete()


  def dumpData(self,outfile):
    '''Store the data in a pickle file.'''

    with open(outfile,'rb') as f:
      pkl.dump({'images':self.data_,'labels':self.labels_},f)




#
# Generate multiple rings and rings of various sizes.
class MultiVariedRingGenerator(RingGenerator):  
  '''Generate image of multiple rings of varying size.
   Ring centers are distributed by a Guassian and the ring
   sizes are uniformly distributed over a range.
   Inherit from RingGenerator
  '''


  def __init__(self,N,n_rings,**kwargs):
    '''Define the multi-varied-ring generation.
      N - Number of points per ring (mean of a Poisson dist.)
      n_rings - Number of rings/image (mean of a Poisson dist.)
      domain - the domain of the image (xy space)
      noise - noise level in image.
      pixels - number of pixels on one side of image (assume square)
      pos - center of Gaussian for ring position distribution
      sig - symmetric width of the position center Gaussian distribution
      r_range - (tuple) range of allowable radii
    '''


    self.N_ = N
    self.nRings_ = n_rings

    self.domain_ = [[-5.,5.],[-5.,5.]]
    self.pixels_ = 500
    self.noise_ = 10000

    self.pos_ = (0.,0.)
    self.sig_ = 3.

    self.r_range_ = (0.25,0.75)
   
    for k,v in kwargs.items():
      if k == 'domain':
        self.domain_ = v
      elif k == 'pixels':
        self.pixels_ = int(v)
      elif k == 'noise':
        self.noise_ = v
      elif k == 'pos':
        self.pos_ = tuple(v)
      elif k == 'sig':
        self.sig_ = float(v)
      elif k == 'r_range':
        self.r_range_ = tuple(v)

    # data is an empty list.
    # it will be populated with flattened images
    self.data_ = []
    self.labels_ = []

  def gen_n_rings(self):
    '''Generate the number of rings in the event, then generate the centers and sizes.'''
    
    n_rings = 0
    while n_rings == 0:
      n_rings = np.random.poisson(self.nRings_)

    # return a set of labels for the event
    # Each label has the following definition
    #  position (x,y), radius
    return list(zip(self.ring_centers(n_rings),self.ring_sizes(n_rings)))

  def ring_centers(self,n_rings):
    '''Generate a set of gaussian ring centers.'''

    return list(np.random.multivariate_normal(self.pos_,np.identity(2)*self.sig_,n_rings))

  def ring_sizes(self,n_rings):
    '''Generate a set of ring radii.'''

    return list(np.random.random(n_rings)*(self.r_range_[1]-self.r_range_[0]) + self.r_range_[0])

  def run(self,nEvents):
    '''Simulate nEvents by parameters already established.'''

    sys.stdout.write('Beginning event generation of %d events.\n' % nEvents)
    #pb = ProgressBar(nEvents)

    self.labels_ = [self.gen_n_rings() for i in range(nEvents)]

    for i in range(nEvents):

      # blank screen
      screen = Screen(self.domain_,self.pixels_)

      for r in range(len(self.labels_[i])):
        points = self.generateRing(self.labels_[i][r][1],self.labels_[i][r][0],self.N_)

        
        points = points[(points[:,0]>self.domain_[0][0])
          & (points[:,0]<self.domain_[0][1])
          & (points[:,1]>self.domain_[1][0])
          & (points[:,1]<self.domain_[1][1])
        ] 
        
        screen.discretize(points) 

      # generate noise
      if self.noise_ is not None:
        npoints = NoiseGenerator().generateNoise(self.noise_,self.domain_)
        screen.discretize(npoints)

      # store image
      self.data_.append(screen.data_.flatten())

      #pb.update(1)

    #pb.complete()


        
  def dumpData(self,outfile):
    '''Store the data in a pickle file.'''

    with open(outfile,'rb') as f:
      pkl.dump({'images':self.data_,'labels':self.labels_},f)

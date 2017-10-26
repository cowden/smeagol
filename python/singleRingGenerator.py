#!/usr/bin/env python2

########################################
# C S Cowden           11 April 2017
########################################



import os,sys
import getopt

import numpy as np
import tables as tb

dir_name = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_name)

import RingGenerator as rg


# parse command line arguments
nSamples = 1000
nHitPerRing = 1000
rRing = 0.5

storeName = 'dataStore.hf5'

seed = 42

opts,args = getopt.getopt(sys.argv[1:],'n:h:r:f:s:',['samples=','ring-hits=','ring-radius=','store=','seed='])

for opt,arg in opts:

  if opt in ('-n','--samples'):
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


# open the file (create it if it does not already exist)
print 'Connecting to data store'
hdfstore = tb.open_file(storeName,mode='a')


# generate rings
srg = rg.SingleRingGenerator(nHitPerRing,rRing)
srg.run(nSamples)


# store the data in the data store
hasNode = False
hasData = False
hasLabels = False
for node in hdfstore.list_nodes('/'):
  if node.name == 'data':
    hasData = True
  elif node.name == 'labels':
    hasLabels = True

hasNode = hasData and hasLabels

if hasNode:
  print 'Appending data to node'
  hdfstore.get_node('/','data').append(srg.data_)
  hdfstore.get_node('/','labels').append(srg.labels_)

else:
  print 'No nodes found ... creating arrays'
  hdfstore.create_earray('/','data',obj=np.array(srg.data_))
  hdfstore.create_earray('/','labels',obj=np.array(srg.labels_))


# close store
hdfstore.close()

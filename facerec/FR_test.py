
import pickle

# coding: utf-8

# # Fine-tuning a Pretrained Network for Style Recognition
# 
# In this example, we'll explore a common approach that is particularly useful in real-world applications: take a pre-trained Caffe network and fine-tune the parameters on your custom data.
# 
# The advantage of this approach is that, since pre-trained networks are learned on a large set of images, the intermediate layers capture the "semantics" of the general visual appearance. Think of it as a very powerful generic visual feature that you can treat as a black box. On top of that, only a relatively small amount of data is needed for good performance on the target task.

# First, we will need to prepare the data. This involves the following parts:
# (1) Get the ImageNet ilsvrc pretrained model with the provided shell scripts.
# (2) Download a subset of the overall Flickr style dataset for this demo.
# (3) Compile the downloaded Flickr dataset into a database that Caffe can then consume.

# In[1]:

caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

import numpy as np
from pylab import *
get_ipython().magic(u'matplotlib inline')
import tempfile
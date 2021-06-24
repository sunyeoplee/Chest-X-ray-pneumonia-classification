import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.math import *
from tensorflow.keras.layers import Layer

import numpy as np

from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve

# delete them 
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

def distance_field(img):

    field = tf.zeros_like(img)
    
    for row in range(len(img)):
        fg_mask = img[row] > 0.5

        if True in fg_mask:
            bg_mask = ~fg_mask

            fg_dist = edt(fg_mask)
            bg_dist = edt(bg_mask)

            new = fg_dist + bg_dist

            new = tf.expand_dims(tf.convert_to_tensor(new, dtype=tf.float32), axis=0)
            indice = tf.expand_dims(tf.constant([row]), axis=1)
            field = tf.tensor_scatter_nd_update(tensor=field, indices=indice, updates=new)
    
    return field


def distance_field(img):
    
    field = np.zeros_like(img)
    
    for row in range(len(img)):
        fg_mask = img[row] > 0.5

        if fg_mask.any():
            bg_mask = ~fg_mask

            fg_dist = edt(fg_mask)
            bg_dist = edt(bg_mask)

            field[row] = fg_dist + bg_dist

    return field

y_true = tf.random.normal((3,3,3,3), 0.5, 0.2)

y_true_dt = tf.numpy_function(distance_field, [y_true], tf.float32)

y = y_true_dt * 2

loss = tf.reduce_mean(y)


y_true[:,:,:,1:] - y_true[:,:,:,:-1]


x = y_true[:,:,1:,:] - y_true[:,:,:-1,:] # horizontal and vertical directions 
y = y_true[:,:,:,1:] - y_true[:,:,:,:-1]

delta_x = x[:,:,1:,:-2]**2
delta_y = y[:,:,:-2,1:]**2
delta_u = K.abs(delta_x + delta_y) 

lenth = K.mean(K.sqrt(delta_u + 0.00000001)) # equ.(11) in the paper

"""
region term
"""

C_1 = np.ones((256, 256))
C_2 = np.zeros((256, 256))

region_in = K.abs(K.mean(y_true[:,0,:,:] * ((y_true[:,0,:,:] - C_1)**2) ) ) # equ.(12) in the paper
region_out = K.abs(K.mean( (1-y_true[:,0,:,:]) * ((y_true[:,0,:,:] - C_2)**2) )) # equ.(12) in the paper

lambdaP = 1 # lambda parameter could be various.
mu = 1 # mu parameter could be various.

return length + lambdaP * (mu * region_in + region_out) 






























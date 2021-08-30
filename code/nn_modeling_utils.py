""" Utility functions for neural network modeling.

"""

import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Dropout,Activation,BatchNormalization,Concatenate
from tensorflow.keras.layers import Dot,RepeatVector,Softmax,Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def bn_dense_block(h,n_nodes):
    """ Dense block with batch-norm
    """
    h = Dense(n_nodes,activation=None)(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    return h


def attention_block(neigh_x,neigh_mask,self_h,h_dim,pad_len):
    """ Attention block
    """
    neigh_h = bn_dense_block(neigh_x,h_dim)
    neigh_self_h = RepeatVector(pad_len)(self_h)
    neigh_self_h = Concatenate()([neigh_h,neigh_self_h])
    e = Dense(1,activation=None)(neigh_self_h)
    e = e+neigh_mask # attention mask
    a = Softmax(axis=1)(e)
    context = Dot(axes=1)([a,neigh_h]) # shape:(n,1,h)
    context = Reshape((-1,))(context) # shape:(n,h)
    return context


def max_pool_block(neigh_x,neigh_mask,h_dim):
    """ Max-pool block
    """
    neigh_h = bn_dense_block(neigh_x,h_dim)
    neigh_h = neigh_h+neigh_mask
    context = tf.math.reduce_max(neigh_h,axis=1)
    return context


def compile_model(self_dim=3,neigh_dim=4,neigh_mask_dim=1,h_dim=100,pad_len=24,optimizer=Adam(lr=0.001),use_dropout=False,use_max_pool=False):
    """ Model implementation - attention network
    """
    self_x = Input(shape=(self_dim))
    self_h = bn_dense_block(self_x,h_dim)
    
    neigh_mask = Input(shape=(pad_len,neigh_mask_dim))
    neigh_x = Input(shape=(pad_len,neigh_dim))
    if use_max_pool:
        neigh_context = max_pool_block(neigh_x,neigh_mask,h_dim)
    else:
        neigh_context = attention_block(neigh_x,neigh_mask,self_h,h_dim,pad_len)
    
    last_neigh_mask = Input(shape=(pad_len,neigh_mask_dim))
    last_neigh_x = Input(shape=(pad_len,neigh_dim))
    if use_max_pool:
        last_neigh_context = max_pool_block(last_neigh_x,last_neigh_mask,h_dim)
    else:
        last_neigh_context = attention_block(last_neigh_x,last_neigh_mask,self_h,h_dim,pad_len)
    
    comb_h = Concatenate(axis=-1)([self_h,neigh_context,last_neigh_context])
    if use_dropout:
        comb_h = Dropout(0.3)(comb_h)
    out = Dense(1,activation=None)(comb_h)

    model = Model(inputs=[self_x,neigh_x,neigh_mask,last_neigh_x,last_neigh_mask],outputs=out)
    model.compile(loss="mse",optimizer=optimizer)
    return model

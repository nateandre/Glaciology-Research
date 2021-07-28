""" General utility functions.

"""

import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error

from data_utils import gather_data


def flatten_matrix(mat):
    """ Flattens matrix and removes invalid values.
    """
    mat = mat.copy()
    mat.shape = (683*701)
    mat = mat[~np.isnan(mat)]
    return mat


def plot_mat_together(mat_1,mat_2,title_1="Summer",title_2="Winter",fsize=12):
    """ Plots two matrices side-by-side.
    """
    plt.figure(figsize=(fsize,fsize))
    plt.subplot(1,2,1)
    plt.title(title_1)
    plt.imshow(mat_1)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title(title_2)
    plt.imshow(mat_2)
    plt.axis('off')
    plt.show()


def plot_mat_hist_together(mat_1,mat_2,title_1="Summer",title_2="Winter",f1=12,f2=4):
    """ Plots histogram of values of two matrices side-by-side.
    """
    mat_1_flat = flatten_matrix(mat_1)
    mat_2_flat = flatten_matrix(mat_2)
    plt.figure(figsize=(f1,f2))
    plt.subplot(1,2,1)
    plt.title(title_1)
    plt.hist(mat_1_flat,bins=70)
    plt.subplot(1,2,2)
    plt.title(title_2)
    plt.hist(mat_2_flat,bins=70)
    plt.show()


def plot_single_mat(mat,title="Combined",fsize=6):
    """ Plots single matrix.
    """
    plt.figure(figsize=(fsize,fsize))
    plt.title(title)
    plt.imshow(mat)
    plt.axis('off')
    plt.show()
    

def plot_single_mat_hist(mat,title="Combined",f1=7,f2=4):
    """ Plots histogram of values of single matrix.
    """
    mat_flat = flatten_matrix(mat)
    plt.figure(figsize=(f1,f2))
    plt.title(title)
    plt.hist(mat_flat,bins=70)
    plt.show()
    

def standard_model_validation(test_year,season,model_dir,pad_len=24,output_dir="../model_output/",data_dir="../data/"):
    """ Returns model performance for standard NN
    """
    _,test_data,_ = gather_data(test_year,season,pad_len,data_dir)
    test_mb,test_features,test_neigh_features,test_last_neigh_features,test_sites,test_neigh_mask,test_last_neigh_mask,_ = test_data

    resids = []
    for test_i in range(0,len(test_sites)):    
        test_x = np.expand_dims(test_features[test_i],0)
        test_neigh_x = np.expand_dims(test_neigh_features[test_i],0)
        test_last_neigh_x = np.expand_dims(test_last_neigh_features[test_i],0)
        test_mask = np.expand_dims(test_neigh_mask[test_i],0)
        test_last_mask = np.expand_dims(test_last_neigh_mask[test_i],0)
        test_y = test_mb[test_i]
        site_i = test_sites[test_i]
        
        site_model = tf.keras.models.load_model(output_dir+model_dir+"/"+site_i)
        test_pred = site_model([test_x,test_neigh_x,test_mask,test_last_neigh_x,test_last_mask])
        resids.append((test_y-float(test_pred))**2)
        
    return math.sqrt(sum(resids)/len(resids))


def standard_model_gp_validation(test_year,season,model_dir,pad_len=24,output_dir="../model_output/",data_dir="../data/"):
    """ Returns model performance for standard NN
    """
    _,test_data,_ = gather_data(test_year,season,pad_len,data_dir)
    test_mb,test_features,test_neigh_features,test_last_neigh_features,test_sites,test_neigh_mask,test_last_neigh_mask,_ = test_data

    resids = []
    for test_i in range(0,len(test_sites)):    
        test_x = np.expand_dims(test_features[test_i],0)
        test_neigh_x = np.expand_dims(test_neigh_features[test_i],0)
        test_last_neigh_x = np.expand_dims(test_last_neigh_features[test_i],0)
        test_mask = np.expand_dims(test_neigh_mask[test_i],0)
        test_last_mask = np.expand_dims(test_last_neigh_mask[test_i],0)
        test_y = test_mb[test_i]
        site_i = test_sites[test_i]
        
        site_model = tf.keras.models.load_model(output_dir+model_dir+"/"+site_i)
        test_pred = site_model([test_x,test_neigh_x,test_mask,test_last_neigh_x,test_last_mask])
        resids.append((test_y-float(test_pred))**2)
        
    return math.sqrt(sum(resids)/len(resids))


def standard_model_gp_validation(test_year,season,model_dir,pad_len=24,output_dir="../model_output/",data_dir="../data/"):
    """ Returns model performance for standard NN+GP
    """
    _,test_data,_ = gather_data(test_year,season,pad_len,data_dir)
    test_mb,test_features,test_neigh_features,test_last_neigh_features,test_sites,test_neigh_mask,test_last_neigh_mask,test_trans_year = test_data

    gp_vars = []
    resids = []
    for test_i in range(0,len(test_sites)):    
        test_x = np.expand_dims(test_features[test_i],0)
        test_neigh_x = np.expand_dims(test_neigh_features[test_i],0)
        test_last_neigh_x = np.expand_dims(test_last_neigh_features[test_i],0)
        test_mask = np.expand_dims(test_neigh_mask[test_i],0)
        test_last_mask = np.expand_dims(test_last_neigh_mask[test_i],0)
        test_tr_year = np.expand_dims(test_trans_year[test_i],0)
        test_y = test_mb[test_i]
        site_i = test_sites[test_i]
        
        site_model = tf.keras.models.load_model(output_dir+model_dir+"/"+site_i)
        embedding_model = Model(inputs=site_model.input,outputs=site_model.layers[-2].output)
        test_emb = embedding_model([test_x,test_neigh_x,test_mask,test_last_neigh_x,test_last_mask]).numpy()
        gp_test_x = np.hstack([test_x,test_tr_year,test_emb])

        m = tf.saved_model.load(output_dir+model_dir+"/gp_models/"+site_i)
        gp_pred_tup = m.predict_y_compiled(gp_test_x)
        gp_pred,gp_var = float(gp_pred_tup[0]),float(gp_pred_tup[1])

        resids.append((test_y-float(gp_pred))**2)
        gp_vars.append(gp_var)
        
    return math.sqrt(sum(resids)/len(resids)),sum(gp_vars)/len(gp_vars)


def standard_model_validation_time(test_year,season,model_dir,pad_len=24,output_dir="../model_output/",data_dir="../data/",model_type="time"):
    """ Returns model performance for time-series NN
    """
    _,test_data,_ = gather_data(test_year,season,pad_len,data_dir,model_type=model_type)
    test_mb,test_features,test_neigh_features,test_last_neigh_features,test_sites,test_neigh_mask,test_last_neigh_mask,_ = test_data

    test_x = test_features
    test_neigh_x = test_neigh_features
    test_last_neigh_x = test_last_neigh_features
    test_mask = test_neigh_mask
    test_last_mask = test_last_neigh_mask
    test_y = test_mb

    site_model = tf.keras.models.load_model(output_dir+model_dir)
    test_pred = site_model([test_x,test_neigh_x,test_mask,test_last_neigh_x,test_last_mask])
    test_sr = mean_squared_error(test_y,test_pred)

    return math.sqrt(test_sr)


def standard_model_gp_validation_time(test_year,season,model_dir,pad_len=24,output_dir="../model_output/",data_dir="../data/",model_type="time"):
    """ Returns model performance for time-series NN+GP
    """
    _,test_data,_ = gather_data(test_year,season,pad_len,data_dir,model_type=model_type)
    test_mb,test_features,test_neigh_features,test_last_neigh_features,test_sites,test_neigh_mask,test_last_neigh_mask,test_trans_year = test_data

    test_x = test_features
    test_neigh_x = test_neigh_features
    test_last_neigh_x = test_last_neigh_features
    test_mask = test_neigh_mask
    test_last_mask = test_last_neigh_mask
    test_y = test_mb
    test_tr_year = test_trans_year

    site_model = tf.keras.models.load_model(output_dir+model_dir)
    embedding_model = Model(inputs=site_model.input,outputs=site_model.layers[-2].output)
    test_emb = embedding_model([test_x,test_neigh_x,test_mask,test_last_neigh_x,test_last_mask]).numpy()
    gp_test_x = np.hstack([test_x,test_tr_year,test_emb])

    m = tf.saved_model.load(output_dir+model_dir+"/gp_models")
    gp_pred_tup = m.predict_y_compiled(gp_test_x)
    gp_pred,gp_var = gp_pred_tup[0].numpy(),gp_pred_tup[1].numpy()
    test_sr = mean_squared_error(test_y,gp_pred)

    return math.sqrt(test_sr),float(np.mean(gp_var))

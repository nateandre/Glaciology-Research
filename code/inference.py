""" Modeling for entire glacier. Inference designed for 2015 to be comparable with other work.

"""

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Model

from data_utils import gather_data_inference,get_glacier_data


def main():
    season = "summer"  # summer/winter
    model_type="standard" # standard/time
    gp_modeling=False
    model_dir = "graph_modeling_3_t_2_full"
    save_dir = "inference"
    pad_len=25


    run_glacier_inference(season,pad_len,model_type,gp_modeling,model_dir,save_dir)
    #combine_matrices(save_dir)


def combine_matrices(save_dir,output_dir="../model_output/"):
    """ Used to combine summer and winter prediction matrices.
    """
    mass_balance_matrix_s = np.load(output_dir+save_dir+"/mass_balance_matrix_s.npy")
    mass_balance_matrix_w = np.load(output_dir+save_dir+"/mass_balance_matrix_w.npy")
    combined_mb_matrix = mass_balance_matrix_s+mass_balance_matrix_w
    np.save(output_dir+save_dir+"/combined_mb_matrix",combined_mb_matrix)


def nn_inference(inference_data,season,save_dir,model_dir,data_dir="../data/",output_dir="../model_output/",batch_size=100):
    """ Inference for deep learning models.
    """
    sc, batch_neigh_x,batch_mask, batch_last_neigh_x,batch_last_mask, _ = inference_data
    g_features,g_indices,mass_balance_matrix,_ = get_glacier_data(sc,data_dir)
    model = tf.keras.models.load_model(output_dir+model_dir)

    # getting the mass balance predictions
    for i in range(0,len(g_indices),batch_size):
        batch_indices = g_indices[i:i+batch_size]
        batch_x = g_features[i:i+batch_size]
        batch_neigh_x = batch_neigh_x[0:len(batch_x)]
        batch_mask = batch_mask[0:len(batch_x)]
        batch_last_neigh_x = batch_last_neigh_x[0:len(batch_x)]
        batch_last_mask = batch_last_mask[0:len(batch_x)]
        
        batch_mb_predictions = model([batch_x,batch_neigh_x,batch_mask,batch_last_neigh_x,batch_last_mask]).numpy()
        
        for j in range(len(batch_indices)):
            this_x,this_y = batch_indices[j]
            mass_balance_matrix[this_x,this_y]=batch_mb_predictions[j]

    fname = "mass_balance_matrix_s"
    if season=="winter":
        fname = "mass_balance_matrix_w"
    np.save(output_dir+save_dir+"/"+fname,mass_balance_matrix)


def gp_inference(inference_data,season,save_dir,model_dir,data_dir="../data/",output_dir="../model_output/",batch_size=100):
    """ Inference for GP models.
    """
    sc, batch_neigh_x,batch_mask, batch_last_neigh_x,batch_last_mask, batch_trans_year = inference_data
    g_features,g_indices,mass_balance_matrix,mb_variance_matrix = get_glacier_data(sc,data_dir)

    site_model = tf.keras.models.load_model(output_dir+model_dir)
    embedding_model = Model(inputs=site_model.input,outputs=site_model.layers[-2].output)
    gp_model = tf.saved_model.load(output_dir+model_dir+"/gp_models")

    # getting the mass balance predictions
    for i in range(0,len(g_indices),batch_size):
        batch_indices = g_indices[i:i+batch_size]
        batch_x = g_features[i:i+batch_size]
        batch_neigh_x = batch_neigh_x[0:len(batch_x)]
        batch_mask = batch_mask[0:len(batch_x)]
        batch_last_neigh_x = batch_last_neigh_x[0:len(batch_x)]
        batch_last_mask = batch_last_mask[0:len(batch_x)]
        batch_trans_year = batch_trans_year[0:len(batch_x)]
        
        batch_embedding = embedding_model([batch_x,batch_neigh_x,batch_mask,batch_last_neigh_x,batch_last_mask]).numpy()
        batch_gp_x = np.hstack([batch_x,batch_trans_year,batch_embedding])
        batch_gp_pred,batch_gp_var = gp_model.predict_y_compiled(batch_gp_x)
        batch_gp_pred,batch_gp_var = batch_gp_pred.numpy(),batch_gp_var.numpy()

        for j in range(len(batch_indices)):
            this_x,this_y = batch_indices[j]
            mass_balance_matrix[this_x,this_y]=batch_gp_pred[j]
            mb_variance_matrix[this_x,this_y]=batch_gp_var[j]

    fname = "mass_balance_matrix_s"
    v_fname = "mb_variance_matrix_s"
    if season=="winter":
        fname = "mass_balance_matrix_w"
        v_fname = "mb_variance_matrix_w"
    np.save(output_dir+save_dir+"/"+fname,mass_balance_matrix)
    np.save(output_dir+save_dir+"/"+v_fname,mb_variance_matrix)


def run_glacier_inference(season,pad_len,model_type,gp_modeling,model_dir,save_dir,test_year=2015,data_dir="../data/",output_dir="../model_output/",batch_size=100):
    """ Runs inference over entire glacier.
    """
    inference_data = gather_data_inference(test_year,season,pad_len,model_type,data_dir=data_dir)

    if save_dir not in os.listdir(output_dir):
        os.mkdir(output_dir+save_dir)

    if gp_modeling == False:
        nn_inference(inference_data,season,save_dir,model_dir,data_dir,output_dir,batch_size)
    else:
        gp_inference(inference_data,season,save_dir,model_dir,data_dir,output_dir,batch_size)


if __name__=="__main__":
    main()

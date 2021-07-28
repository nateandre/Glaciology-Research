""" Same-year modeling. No validation.

"""

import tensorflow as tf
import numpy as np
import math
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

from nn_modeling_utils import compile_model
from data_utils import gather_data


def main():
    test_year = 2015
    season = "summer"  # summer/winter
    model_dir = "graph_modeling_3_t_2_full"
    pad_len=25
    run_simulation(test_year,season,model_dir,pad_len)


def run_simulation(test_year,season,model_dir,pad_len=24,data_dir="../data/",output_dir="../model_output/"):
    """ Trains the model
    """
    train_data,test_data,years = gather_data(test_year,season,pad_len,data_dir)
    all_val_rmse,site_saved_models = train_model(train_data,test_data,years,test_year,model_dir,pad_len,output_dir=output_dir)
    # Model val average:
    print("Avg:",round(sum(all_val_rmse)/len(all_val_rmse),4))
    # Indiv model performance
    print("Ind:",site_saved_models)


def train_model(train_data,test_data,years,test_year,model_dir,pad_len,epochs=100,num_inits=5,n_val_same=10,n_val_other=30,batch_size=25,output_dir="../model_output/"):
    """ Used to train the deep learning model
    args:
        model_dir (str): specifies directory name for storing this model
        output_dir (str): specifies directory to store trained models
    """
    all_val_rmse = []

    train_mb,train_features,train_neigh_features,train_last_neigh_features,train_sites,train_neigh_mask,train_last_neigh_mask,_ = train_data
    test_mb,test_features,test_neigh_features,test_last_neigh_features,test_sites,test_neigh_mask,test_last_neigh_mask,_ = test_data

    # getting training, validation, and testing data:
    train_x = np.vstack([train_features,test_features])
    train_neigh_x = np.vstack([train_neigh_features,test_neigh_features])
    train_last_neigh_x = np.vstack([train_last_neigh_features,test_last_neigh_features])
    train_mask = np.vstack([train_neigh_mask,test_neigh_mask])
    train_last_mask = np.vstack([train_last_neigh_mask,test_last_neigh_mask])
    train_y = np.concatenate([train_mb,test_mb])
    train_years = np.concatenate([years[years!=test_year],years[years==test_year]])
    train_x,train_y,train_neigh_x,train_mask,train_years,train_last_neigh_x,train_last_mask = shuffle(train_x,train_y,train_neigh_x,train_mask,train_years,train_last_neigh_x,train_last_mask,random_state=10)

    val_x_same,val_neigh_x_same,val_y_same,val_mask_same,val_last_neigh_x_same,val_last_mask_same = train_x[train_years==test_year][:n_val_same],train_neigh_x[train_years==test_year][:n_val_same],train_y[train_years==test_year][:n_val_same],train_mask[train_years==test_year][:n_val_same],train_last_neigh_x[train_years==test_year][:n_val_same],train_last_mask[train_years==test_year][:n_val_same]
    val_x_other,val_neigh_x_other,val_y_other,val_mask_other,val_last_neigh_x_other,val_last_mask_other = train_x[train_years!=test_year][:n_val_other],train_neigh_x[train_years!=test_year][:n_val_other],train_y[train_years!=test_year][:n_val_other],train_mask[train_years!=test_year][:n_val_other],train_last_neigh_x[train_years!=test_year][:n_val_other],train_last_mask[train_years!=test_year][:n_val_other]

    train_x_same,train_neigh_x_same,train_y_same,train_mask_same,train_last_neigh_x_same,train_last_mask_same = train_x[train_years==test_year][n_val_same:],train_neigh_x[train_years==test_year][n_val_same:],train_y[train_years==test_year][n_val_same:],train_mask[train_years==test_year][n_val_same:],train_last_neigh_x[train_years==test_year][n_val_same:],train_last_mask[train_years==test_year][n_val_same:]
    train_x_other,train_neigh_x_other,train_y_other,train_mask_other,train_last_neigh_x_other,train_last_mask_other = train_x[train_years!=test_year][n_val_other:],train_neigh_x[train_years!=test_year][n_val_other:],train_y[train_years!=test_year][n_val_other:],train_mask[train_years!=test_year][n_val_other:],train_last_neigh_x[train_years!=test_year][n_val_other:],train_last_mask[train_years!=test_year][n_val_other:]
    train_x,train_neigh_x,train_y,train_mask,train_last_mask,train_last_neigh_x = np.vstack([train_x_same,train_x_other]),np.vstack([train_neigh_x_same,train_neigh_x_other]),np.concatenate([train_y_same,train_y_other]),np.vstack([train_mask_same,train_mask_other]),np.vstack([train_last_mask_same,train_last_mask_other]),np.vstack([train_last_neigh_x_same,train_last_neigh_x_other])

    # get validation metrics:
    run_best_val_rmse = float("inf")

    for _ in range(num_inits):
        best_val_rmse = float("inf")

        # train the model:
        model = compile_model(pad_len=pad_len)
        for epoch_i in range(epochs):
            train_x,train_neigh_x,train_y,train_mask,train_last_neigh_x,train_last_mask = shuffle(train_x,train_neigh_x,train_y,train_mask,train_last_neigh_x,train_last_mask,random_state=10)
            batch_losses = []
            for i in range(0,(len(train_x)//batch_size)*batch_size,batch_size):
                batch_train_x,batch_train_neigh_x,batch_train_y,batch_train_mask,batch_train_last_neigh_x,batch_train_last_mask = train_x[i:i+batch_size],train_neigh_x[i:i+batch_size],train_y[i:i+batch_size],train_mask[i:i+batch_size],train_last_neigh_x[i:i+batch_size],train_last_mask[i:i+batch_size]
                batch_loss = model.train_on_batch([batch_train_x,batch_train_neigh_x,batch_train_mask,batch_train_last_neigh_x,batch_train_last_mask],batch_train_y)
                batch_losses.append(float(batch_loss))
            batch_loss = sum(batch_losses)/len(batch_losses)

            val_pred_same = model([val_x_same,val_neigh_x_same,val_mask_same,val_last_neigh_x_same,val_last_mask_same]).numpy()
            val_sr_same = mean_squared_error(val_y_same,val_pred_same)
            val_rmse_same = math.sqrt(val_sr_same)

            val_pred_other = model([val_x_other,val_neigh_x_other,val_mask_other,val_last_neigh_x_other,val_last_mask_other]).numpy()
            val_sr_other = mean_squared_error(val_y_other,val_pred_other)
            val_rmse_other = math.sqrt(val_sr_other)

            val_rmse = (val_rmse_other+val_rmse_same)/2

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse

            if val_rmse < run_best_val_rmse: # saving the model w/ the lowest validation error
                run_best_val_rmse = val_rmse
                site_saved_models = round(best_val_rmse,6)
                model.save(output_dir+model_dir)
                
        all_val_rmse.append(best_val_rmse)

    return all_val_rmse,site_saved_models


if __name__=="__main__":
    main()

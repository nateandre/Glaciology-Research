""" Time-series modeling.

"""

import tensorflow as tf
import numpy as np
import math
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

from nn_modeling_utils import compile_model
from data_utils import gather_data


def main():
    test_year = 2014
    season = "summer"  # summer/winter
    model_dir = "graph_modeling_s_2014_time_mp"
    pad_len=24
    print(test_year,season)
    run_simulation(test_year,season,model_dir,pad_len,use_max_pool=True)


def run_simulation(test_year,season,model_dir,pad_len=24,data_dir="../data/",output_dir="../model_output/",use_max_pool=False):
    """ Trains the model
    """
    train_data,test_data,years = gather_data(test_year,season,pad_len,data_dir,model_type="time")
    this_test_sr,site_saved_models = train_model(train_data,test_data,years,test_year,model_dir,pad_len,use_max_pool=use_max_pool,output_dir=output_dir)
    # Model averages over inits
    print("Avg:",math.sqrt(round(sum(this_test_sr)/len(this_test_sr),4)))
    # Indiv model performance
    print("Ind:",site_saved_models[1])


def train_model(train_data,test_data,years,test_year,model_dir,pad_len,epochs=100,num_inits=5,n_val=30,batch_size=25,h_dim=100,use_max_pool=False,output_dir="../model_output/"):
    """ Used to train the deep learning model
    args:
        model_dir (str): specifies directory name for storing this model
        output_dir (str): specifies directory to store trained models
    """
    train_mb,train_features,train_neigh_features,train_last_neigh_features,train_sites,train_neigh_mask,train_last_neigh_mask,_ = train_data
    test_mb,test_features,test_neigh_features,test_last_neigh_features,test_sites,test_neigh_mask,test_last_neigh_mask,_ = test_data

    if use_max_pool:
        train_neigh_mask = np.stack([np.squeeze(train_neigh_mask) for _ in range(h_dim)],axis=-1)
        train_last_neigh_mask = np.stack([np.squeeze(train_last_neigh_mask) for _ in range(h_dim)],axis=-1)
        test_neigh_mask = np.stack([np.squeeze(test_neigh_mask) for _ in range(h_dim)],axis=-1)
        test_last_neigh_mask = np.stack([np.squeeze(test_last_neigh_mask) for _ in range(h_dim)],axis=-1)

    # getting training, validation, and testing data:
    train_x = train_features
    train_neigh_x = train_neigh_features
    train_last_neigh_x = train_last_neigh_features
    train_mask = train_neigh_mask
    train_last_mask = train_last_neigh_mask
    train_y = train_mb
    train_years = years[years!=test_year]
    train_x,train_y,train_neigh_x,train_mask,train_years,train_last_neigh_x,train_last_mask = shuffle(train_x,train_y,train_neigh_x,train_mask,train_years,train_last_neigh_x,train_last_mask,random_state=10)

    test_x = test_features
    test_neigh_x = test_neigh_features
    test_last_neigh_x = test_last_neigh_features
    test_mask = test_neigh_mask
    test_last_mask = test_last_neigh_mask
    test_y = test_mb

    val_x,val_neigh_x,val_y,val_mask,val_last_neigh_x,val_last_mask = train_x[:n_val],train_neigh_x[:n_val],train_y[:n_val],train_mask[:n_val],train_last_neigh_x[:n_val],train_last_mask[:n_val]
    train_x,train_neigh_x,train_y,train_mask,train_last_neigh_x,train_last_mask = train_x[n_val:],train_neigh_x[n_val:],train_y[n_val:],train_mask[n_val:],train_last_neigh_x[n_val:],train_last_mask[n_val:]

    # get validation metrics:
    this_test_sr = [] # saving the squared residuals for each init
    run_best_val_rmse = float("inf")

    for _ in range(num_inits):
        best_val_rmse = float("inf")
        best_test_sr = 0
        best_test_rmse = 0

        # train the model:
        model = compile_model(pad_len=pad_len,neigh_mask_dim=train_neigh_mask.shape[-1],use_max_pool=use_max_pool,use_dropout=True)
        for epoch_i in range(epochs):
            train_x,train_neigh_x,train_y,train_mask,train_last_neigh_x,train_last_mask = shuffle(train_x,train_neigh_x,train_y,train_mask,train_last_neigh_x,train_last_mask,random_state=10)
            batch_losses = []
            for i in range(0,(len(train_x)//batch_size)*batch_size,batch_size):
                batch_train_x,batch_train_neigh_x,batch_train_y,batch_train_mask,batch_train_last_neigh_x,batch_train_last_mask = train_x[i:i+batch_size],train_neigh_x[i:i+batch_size],train_y[i:i+batch_size],train_mask[i:i+batch_size],train_last_neigh_x[i:i+batch_size],train_last_mask[i:i+batch_size]
                batch_loss = model.train_on_batch([batch_train_x,batch_train_neigh_x,batch_train_mask,batch_train_last_neigh_x,batch_train_last_mask],batch_train_y)
                batch_losses.append(float(batch_loss))
            batch_loss = sum(batch_losses)/len(batch_losses)

            val_pred = model([val_x,val_neigh_x,val_mask,val_last_neigh_x,val_last_mask]).numpy()
            val_sr = mean_squared_error(val_y,val_pred)
            val_rmse = math.sqrt(val_sr)

            test_pred = model([test_x,test_neigh_x,test_mask,test_last_neigh_x,test_last_mask]).numpy()
            test_sr = mean_squared_error(test_y,test_pred)
            test_rmse = math.sqrt(test_sr)
        
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_test_sr = test_sr
                best_test_rmse = test_rmse

            if val_rmse < run_best_val_rmse: # saving the model w/ the lowest validation error
                run_best_val_rmse = val_rmse
                site_saved_models = (round(best_val_rmse,6),round(best_test_rmse,6))
                model.save(output_dir+model_dir)

        this_test_sr.append(best_test_sr)

    return this_test_sr,site_saved_models
    

if __name__=="__main__":
    main()

""" Same-year modeling.

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
    season = "winter"  # summer/winter
    model_dir = "graph_modeling_3_t_w_2_2014"
    pad_len=24
    run_simulation(test_year,season,model_dir,pad_len)


def run_simulation(test_year,season,model_dir,pad_len=24,data_dir="../data/",output_dir="../model_output/"):
    """ Trains the model
    """
    train_data,test_data,years = gather_data(test_year,season,pad_len,data_dir)
    squared_residuals,site_saved_models = train_model(train_data,test_data,years,test_year,model_dir,pad_len,output_dir=output_dir)
    # Model averages over inits
    print("Avg:",math.sqrt(sum(squared_residuals)/len(squared_residuals)))
    # Indiv model performance
    sq_resid = {}
    for key in site_saved_models:
        sq_resid[key] = list(site_saved_models[key])
        sq_resid[key][1] = sq_resid[key][1]**2
    all_resids = [sq_resid[key][1] for key in sq_resid]
    print("Ind:",math.sqrt(sum(all_resids)/len(all_resids)))


def train_model(train_data,test_data,years,test_year,model_dir,pad_len,epochs=100,num_inits=5,n_val_same=10,n_val_other=30,batch_size=25,output_dir="../model_output/"):
    """ Used to train the deep learning model
    args:
        model_dir (str): specifies directory name for storing this model
        output_dir (str): specifies directory to store trained models
    """
    squared_residuals = []
    site_saved_models = {}

    train_mb,train_features,train_neigh_features,train_last_neigh_features,train_sites,train_neigh_mask,train_last_neigh_mask,_ = train_data
    test_mb,test_features,test_neigh_features,test_last_neigh_features,test_sites,test_neigh_mask,test_last_neigh_mask,_ = test_data

    for test_i in range(0,len(test_sites)):
        # getting training, validation, and testing data:
        add_train_i = [i for i in list(range(0,test_i))+list(range(test_i+1,len(test_sites)))]
        train_x = np.vstack([train_features,test_features[add_train_i]])
        train_neigh_x = np.vstack([train_neigh_features,test_neigh_features[add_train_i]])
        train_last_neigh_x = np.vstack([train_last_neigh_features,test_last_neigh_features[add_train_i]])
        train_mask = np.vstack([train_neigh_mask,test_neigh_mask[add_train_i]])
        train_last_mask = np.vstack([train_last_neigh_mask,test_last_neigh_mask[add_train_i]])
        train_y = np.concatenate([train_mb,test_mb[add_train_i]])
        train_years = np.concatenate([years[years!=test_year],years[years==test_year][:-1]])
        train_x,train_y,train_neigh_x,train_mask,train_years,train_last_neigh_x,train_last_mask = shuffle(train_x,train_y,train_neigh_x,train_mask,train_years,train_last_neigh_x,train_last_mask,random_state=10)

        test_x = np.expand_dims(test_features[test_i],0)
        test_neigh_x = np.expand_dims(test_neigh_features[test_i],0)
        test_last_neigh_x = np.expand_dims(test_last_neigh_features[test_i],0)
        test_mask = np.expand_dims(test_neigh_mask[test_i],0)
        test_last_mask = np.expand_dims(test_last_neigh_mask[test_i],0)
        test_y = test_mb[test_i]
        site_i = test_sites[test_i]
        
        val_x_same,val_neigh_x_same,val_y_same,val_mask_same,val_last_neigh_x_same,val_last_mask_same = train_x[train_years==test_year][:n_val_same],train_neigh_x[train_years==test_year][:n_val_same],train_y[train_years==test_year][:n_val_same],train_mask[train_years==test_year][:n_val_same],train_last_neigh_x[train_years==test_year][:n_val_same],train_last_mask[train_years==test_year][:n_val_same]
        val_x_other,val_neigh_x_other,val_y_other,val_mask_other,val_last_neigh_x_other,val_last_mask_other = train_x[train_years!=test_year][:n_val_other],train_neigh_x[train_years!=test_year][:n_val_other],train_y[train_years!=test_year][:n_val_other],train_mask[train_years!=test_year][:n_val_other],train_last_neigh_x[train_years!=test_year][:n_val_other],train_last_mask[train_years!=test_year][:n_val_other]
    
        train_x_same,train_neigh_x_same,train_y_same,train_mask_same,train_last_neigh_x_same,train_last_mask_same = train_x[train_years==test_year][n_val_same:],train_neigh_x[train_years==test_year][n_val_same:],train_y[train_years==test_year][n_val_same:],train_mask[train_years==test_year][n_val_same:],train_last_neigh_x[train_years==test_year][n_val_same:],train_last_mask[train_years==test_year][n_val_same:]
        train_x_other,train_neigh_x_other,train_y_other,train_mask_other,train_last_neigh_x_other,train_last_mask_other = train_x[train_years!=test_year][n_val_other:],train_neigh_x[train_years!=test_year][n_val_other:],train_y[train_years!=test_year][n_val_other:],train_mask[train_years!=test_year][n_val_other:],train_last_neigh_x[train_years!=test_year][n_val_other:],train_last_mask[train_years!=test_year][n_val_other:]
        train_x,train_neigh_x,train_y,train_mask,train_last_mask,train_last_neigh_x = np.vstack([train_x_same,train_x_other]),np.vstack([train_neigh_x_same,train_neigh_x_other]),np.concatenate([train_y_same,train_y_other]),np.vstack([train_mask_same,train_mask_other]),np.vstack([train_last_mask_same,train_last_mask_other]),np.vstack([train_last_neigh_x_same,train_last_neigh_x_other])
    
        # get validation metrics:
        this_test_sr = [] # saving the squared residuals for each init
        run_best_val_rmse = float("inf")

        for _ in range(num_inits):
            best_val_rmse = float("inf")
            best_test_sr = 0
            best_test_rmse = 0
        
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
                
                test_pred = float(model([test_x,test_neigh_x,test_mask,test_last_neigh_x,test_last_mask]))
                test_sr = mean_squared_error([test_y],[test_pred])
                test_rmse = math.sqrt(test_sr)

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_test_sr = test_sr
                    best_test_rmse = test_rmse
                    
                if val_rmse < run_best_val_rmse: # saving the model w/ the lowest validation error
                    run_best_val_rmse = val_rmse
                    site_saved_models[site_i] = (round(best_val_rmse,6),round(best_test_rmse,6))
                    model.save(output_dir+model_dir+"/"+site_i)
                    
            this_test_sr.append(best_test_sr)
        
        this_avg_sr = round(sum(this_test_sr)/len(this_test_sr),4)
        squared_residuals.append(this_avg_sr)
        print("---",site_i, ";", this_avg_sr, ";", this_test_sr)


    return squared_residuals,site_saved_models


if __name__=="__main__":
    main()

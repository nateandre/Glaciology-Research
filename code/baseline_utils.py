""" Baseline utility functions.

"""

import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import math
from data_utils import gather_data


def run_model_standard(base_model,num_inits,train_mb,train_features,test_mb,test_features,test_sites):
    """ Gets performance metric for standard model - using cross-val
    """
    squared_residuals = []
    
    for test_i in range(0,len(test_sites)):
        add_train_i = [i for i in list(range(0,test_i))+list(range(test_i+1,len(test_sites)))]
        train_x = np.vstack([train_features,test_features[add_train_i]])
        train_y = np.concatenate([train_mb,test_mb[add_train_i]])
        train_x,train_y = shuffle(train_x,train_y,random_state=10)
        test_x = np.expand_dims(test_features[test_i],0)
        test_y = test_mb[test_i]
        site_i = test_sites[test_i]

        this_test_sr = [] # saving the squared residuals for each init
        run_best_val_rmse = float("inf")

        for _ in range(num_inits):
            model = sklearn.base.clone(base_model)
            model.fit(train_x,train_y)
            test_pred = float(model.predict(test_x))
            test_sr = mean_squared_error([test_y],[test_pred])
            this_test_sr.append(test_sr)
        this_avg_sr = sum(this_test_sr)/len(this_test_sr)
        squared_residuals.append(this_avg_sr)
        
    return math.sqrt(sum(squared_residuals)/len(squared_residuals))


def run_model_time(base_model,num_inits,train_mb,train_features,test_mb,test_features,test_sites):
    """ Gets performance metric for time-series model
    """
    squared_residuals = []
    
    for _ in range(num_inits):
        model = sklearn.base.clone(base_model)
        model.fit(train_features,train_mb)
        test_pred = model.predict(test_features)
        test_sr = mean_squared_error(test_mb,test_pred)
        squared_residuals.append(test_sr)
        
    return math.sqrt(sum(squared_residuals)/len(squared_residuals))


def print_results(output):
    """ Prints output
    """
    for model_type in output:
        print("----"+model_type+":")
        for year in output[model_type]:
            print("{}: summer:{:.6f}, winter:{:.6f}".format(year,output[model_type][year]['summer'],output[model_type][year]['winter']))


def run_linear_regression(model_parameters,model_type,train_mb,train_features,test_mb,test_features,test_sites,num_inits=1):
    """ Instantiates and runs linear regression model
    """
    model = LinearRegression()
    if model_type=="time":
        test_metric = run_model_time(model,num_inits,train_mb,train_features,test_mb,test_features,test_sites)
    elif model_type=="standard":
        test_metric = run_model_standard(model,num_inits,train_mb,train_features,test_mb,test_features,test_sites)
    return test_metric,None


def run_random_forest(model_parameters,model_type,train_mb,train_features,test_mb,test_features,test_sites,num_inits=3):
    """ Instantiates and runs random-forest model
    """
    best_parameters = None
    best_test_metric = 1000
    
    for n_estimators in model_parameters['n_estimators']:
        for max_depth in model_parameters['max_depth']:
            model = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth)
            if model_type=="time":
                test_metric = run_model_time(model,num_inits,train_mb,train_features,test_mb,test_features,test_sites)
            elif model_type=="standard":
                test_metric = run_model_standard(model,num_inits,train_mb,train_features,test_mb,test_features,test_sites)
            
            if test_metric<best_test_metric:
                best_test_metric=test_metric
                best_parameters=(n_estimators,max_depth)
                
    return best_test_metric,best_parameters


def run_gradient_boosting(model_parameters,model_type,train_mb,train_features,test_mb,test_features,test_sites,num_inits=3):
    """ Instantiates and runs gradient-boosting model
    """
    best_parameters = None
    best_test_metric = 1000
    
    for n_estimators in model_parameters['n_estimators']:
        for max_depth in model_parameters['max_depth']:
            model = GradientBoostingRegressor(n_estimators=n_estimators,max_depth=max_depth)
            if model_type=="time":
                test_metric = run_model_time(model,num_inits,train_mb,train_features,test_mb,test_features,test_sites)
            elif model_type=="standard":
                test_metric = run_model_standard(model,num_inits,train_mb,train_features,test_mb,test_features,test_sites)
            
            if test_metric<best_test_metric:
                best_test_metric=test_metric
                best_parameters=(n_estimators,max_depth)
                
    return best_test_metric,best_parameters


def run_dnn(model_parameters,model_type,train_mb,train_features,test_mb,test_features,test_sites,num_inits=3):
    """ Instantiates and runs random-forest model
    """
    best_parameters = None
    best_test_metric = 1000
    
    for hidden_layer_sizes in model_parameters['hidden_layer_sizes']:
        for alpha in model_parameters['alpha']:
            model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,alpha=alpha,early_stopping=True,max_iter=1000)
            if model_type=="time":
                test_metric = run_model_time(model,num_inits,train_mb,train_features,test_mb,test_features,test_sites)
            elif model_type=="standard":
                test_metric = run_model_standard(model,num_inits,train_mb,train_features,test_mb,test_features,test_sites)
            
            if test_metric<best_test_metric:
                best_test_metric=test_metric
                best_parameters=(hidden_layer_sizes,alpha)
                
    return best_test_metric,best_parameters


def validate_model(model_name,model_parameters):
    """ Gets performance metrics for baseline models
    """
    test_results={}
    test_params={}
    for model_type in ["standard","time"]:
        test_results[model_type]={}
        test_params[model_type]={}
        for test_year in [2014,2015]:
            test_results[model_type][test_year]={}
            test_params[model_type][test_year]={}
            for season in ["winter","summer"]:         
                train_data,test_data,years = gather_data(test_year,season,pad_len=24,data_dir="../data/",model_type=model_type)
                train_mb,train_features,_,_,_,_,_,train_years = train_data
                test_mb,test_features,_,_,test_sites,_,_,test_years = test_data
                train_features = np.hstack([train_features,train_years])
                test_features = np.hstack([test_features,test_years])
                
                if model_name=="linear_regression":
                    test_metric,best_params = run_linear_regression(model_parameters,model_type,train_mb,train_features,test_mb,test_features,test_sites)
                elif model_name=="random_forest":
                    test_metric,best_params = run_random_forest(model_parameters,model_type,train_mb,train_features,test_mb,test_features,test_sites)
                elif model_name=="gradient_boosting":
                    test_metric,best_params = run_gradient_boosting(model_parameters,model_type,train_mb,train_features,test_mb,test_features,test_sites)
                elif model_name=="dnn":
                    test_metric,best_params = run_dnn(model_parameters,model_type,train_mb,train_features,test_mb,test_features,test_sites)
                
                test_results[model_type][test_year][season] = test_metric
                test_params[model_type][test_year][season] = best_params
                
    return test_results,test_params

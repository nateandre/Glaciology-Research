""" Same-year GP modeling. No validation.

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import os
import math
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import gpflow
from gpflow.ci_utils import ci_niter
f64 = gpflow.utilities.to_default_float

from gp_modeling_utils import Linear
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
    train_model(train_data,test_data,years,test_year,model_dir,output_dir=output_dir) # no metrics


def train_model(train_data,test_data,years,test_year,model_dir,output_dir="../model_output/"):
    """ Used to train the GP component on top of the deep learning model
    args:
        model_dir (str): specifies directory name for storing this model
        output_dir (str): specifies directory to store trained models
    """
    train_mb,train_features,train_neigh_features,train_last_neigh_features,train_sites,train_neigh_mask,train_last_neigh_mask,train_trans_year = train_data
    test_mb,test_features,test_neigh_features,test_last_neigh_features,test_sites,test_neigh_mask,test_last_neigh_mask,test_trans_year = test_data

    # getting training, validation, and testing data:
    train_x = np.vstack([train_features,test_features])
    train_neigh_x = np.vstack([train_neigh_features,test_neigh_features])
    train_last_neigh_x = np.vstack([train_last_neigh_features,test_last_neigh_features])
    train_mask = np.vstack([train_neigh_mask,test_neigh_mask])
    train_last_mask = np.vstack([train_last_neigh_mask,test_last_neigh_mask])
    train_y = np.expand_dims(np.concatenate([train_mb,test_mb]),axis=-1)
    train_tr_year = np.vstack([train_trans_year,test_trans_year])

    site_model = keras.models.load_model(output_dir+model_dir)
    b_ = site_model.trainable_variables[-1].numpy()
    A_ = site_model.trainable_variables[-2].numpy()
    embedding_model = Model(inputs=site_model.input,outputs=site_model.layers[-2].output)
        
    train_emb = embedding_model([train_x,train_neigh_x,train_mask,train_last_neigh_x,train_last_mask]).numpy()
    gp_train_x = np.hstack([train_x,train_tr_year,train_emb])
        
    spatial_k = gpflow.kernels.Matern52(active_dims=[0,1,2])
    time_k = gpflow.kernels.Matern52(active_dims=[3])
    kernel = spatial_k*time_k

    lags = list(range(1,101))

    num_retries = 0
    while True:
        m = gpflow.models.GPR(data=(gp_train_x,train_y),kernel=kernel,mean_function=Linear(A=A_,b=b_))
        m.kernel.kernels[0].variance.assign(10.0)
        m.kernel.kernels[1].variance.assign(10.0)
        m.kernel.kernels[0].lengthscales.assign(1.0)
        m.kernel.kernels[1].lengthscales.assign(1.0)
        m.likelihood.variance.assign(0.01)
        gpflow.set_trainable(m.mean_function.A,False)
        gpflow.set_trainable(m.mean_function.b,False)
        gpflow.set_trainable(m.likelihood.variance,False)
        
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

        m.kernel.kernels[0].variance.prior = tfd.Gamma(f64(1.0),f64(4.0))
        m.kernel.kernels[1].variance.prior = tfd.Gamma(f64(1.0),f64(4.0))
        m.kernel.kernels[0].lengthscales.prior = tfd.Gamma(f64(1.0),f64(4.0))
        m.kernel.kernels[1].lengthscales.prior = tfd.Gamma(f64(1.0),f64(4.0))

        if num_retries < 20:
            num_burnin_steps = ci_niter(2000) # number of steps prior to getting the samples
        else:
            num_burnin_steps = ci_niter(4000)
        num_samples = ci_niter(1000)

        hmc_helper = gpflow.optimizers.SamplingHelper(
            m.log_posterior_density, m.trainable_parameters
        )
        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=0.01
        )
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            hmc, num_adaptation_steps=100, target_accept_prob=f64(0.75), adaptation_rate=0.1
        )
        @tf.function
        def run_chain_fn():
            return tfp.mcmc.sample_chain(
                num_results=num_samples,
                num_burnin_steps=num_burnin_steps,
                current_state=hmc_helper.current_state,
                kernel=adaptive_hmc,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
            )
        samples, traces = run_chain_fn()
        parameter_samples = hmc_helper.convert_to_constrained_values(samples)

        m.kernel.kernels[0].lengthscales.assign(float(np.mean(parameter_samples[0])))
        m.kernel.kernels[0].variance.assign(float(np.mean(parameter_samples[1])))
        m.kernel.kernels[1].lengthscales.assign(float(np.mean(parameter_samples[2])))
        m.kernel.kernels[1].variance.assign(float(np.mean(parameter_samples[3])))
        
        # testing if mcmc converged, based on autocorrelation
        all_param_series = []
        for i in range(len(parameter_samples)):
            param_series = pd.Series(parameter_samples[i])
            auto_corrs = []
            for l in lags:
                auto_corr = param_series.autocorr(l)
                auto_corrs.append(auto_corr)
            all_param_series.append(auto_corrs)
            
        converged = True
        high = 0
        num = 0
        for c_i,corrs in enumerate(all_param_series):
            corr = corrs[9]
            if abs(corr) > abs(high):
                high = corr
                num = c_i+1
        print("{},{}; ".format(num,round(high,2)),end='',flush=True)
        
        if abs(high)<=0.1:
            print()
            break
        num_retries += 1
        
        
    # saving the sampled parameter values:
    if "gp_samples" not in os.listdir(output_dir+model_dir):
        os.mkdir(output_dir+model_dir+"/gp_samples")
    all_parameter_samples = np.stack(parameter_samples)
    np.save(output_dir+model_dir+"/gp_samples/parameters",all_parameter_samples)
    
    # saving the GP model:
    m.predict_y_compiled = tf.function(
        m.predict_y,input_signature=[tf.TensorSpec(shape=[None,304],dtype=tf.float64)]
    )
    tf.saved_model.save(m,output_dir+model_dir+"/gp_models")


if __name__=="__main__":
    main()    

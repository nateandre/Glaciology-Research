""" Utility functions for data.

"""

import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(test_year,season,data_dir="../data/"):
    """ Loads the training data
    args:
        test_year (int): specifies the year being used as the test year
        season (str): winter/summer, determines which labels are loaded
    """
    if season == 'summer':
        mb_y = np.load(data_dir+"mb_s.npy")
    elif season == 'winter':
        mb_y = np.load(data_dir+"mb_w.npy")

    sites = np.load(data_dir+"sites.npy",allow_pickle=True)
    features = np.load(data_dir+"features.npy") # x,y,z,t

    years = features[:,-1]
    features = features[years<=test_year]
    mb_y = mb_y[years<=test_year]
    sites = sites[years<=test_year]
    years = years[years<=test_year]

    sc = StandardScaler()
    trans_years = sc.fit_transform(np.expand_dims(features[:,-1],axis=-1)) # used for the GP component
    trans_features = sc.fit_transform(np.hstack([features[:,:-1],np.expand_dims(mb_y,axis=-1)])) # includes the label
    features = trans_features[:,:-1] # excludes the label

    return features,trans_features,mb_y,trans_years,sites,years

    
def generate_traditional_neighborhood_features(trans_features,sites,years,pad_len,earliest_year=1997):
    """ Generates the neighborhood features for each node - this includes all neighboring nodes for the test year and all nodes from the year prior to the test year
    args:
        pad_len (int): determines the amount of padding for the site neighborhood
        earliest_year (int): earliest year in the dataset
    """
    all_indices = np.array([i for i in range(len(trans_features))])
    all_neighbor_features = []
    all_last_neighbor_features = []
    all_neighbor_mask = [] # mask required for the attention mechanism
    all_last_neighbor_mask = []

    for i,year in enumerate(years):
        this_site = sites[i]
        this_sites = sites[years==year]
        this_indices = all_indices[years==year]
        neigh_indices = this_indices[this_sites!=this_site] # remove the test site from being included as a feature

        this_neigh_mask = np.zeros((pad_len))
        this_neigh_mask[len(neigh_indices):]=-np.inf
        all_neighbor_mask.append(this_neigh_mask)
        neigh_feat = trans_features[neigh_indices]
        neigh_feat = np.vstack([neigh_feat,np.zeros((pad_len-len(neigh_feat),neigh_feat.shape[-1]))])
        all_neighbor_features.append(neigh_feat)
    
        if year != earliest_year: # earliest year in dataset
            last_neigh_indices = all_indices[years==year-1]
            last_neigh_mask = np.zeros((pad_len))
            last_neigh_mask[len(last_neigh_indices):]=-np.inf
            last_neigh_feat = trans_features[last_neigh_indices]
            last_neigh_feat = np.vstack([last_neigh_feat,np.zeros((pad_len-len(last_neigh_feat),last_neigh_feat.shape[-1]))])
        else:
            last_neigh_mask = np.zeros((pad_len))
            last_neigh_feat = np.zeros((pad_len,neigh_feat.shape[-1]))
            
        all_last_neighbor_mask.append(last_neigh_mask)
        all_last_neighbor_features.append(last_neigh_feat)
    
    all_neighbor_features = np.stack(all_neighbor_features)
    all_last_neighbor_features = np.stack(all_last_neighbor_features)
    all_neighbor_mask = np.expand_dims(np.stack(all_neighbor_mask),axis=-1)
    all_last_neighbor_mask = np.expand_dims(np.stack(all_last_neighbor_mask),axis=-1)

    return all_neighbor_features,all_neighbor_mask, all_last_neighbor_features,all_last_neighbor_mask


def generate_time_series_neighborhood_features(trans_features,sites,years,pad_len,earliest_year=1997):
    """ Generates the neighborhood features for each node - these are simply the node features from the previous 2 years prior to the test year
    """
    all_indices = np.array([i for i in range(len(trans_features))])
    all_neighbor_features = []
    all_last_neighbor_features = []
    all_neighbor_mask = [] # mask required for the attention mechanism
    all_last_neighbor_mask = []

    for i,year in enumerate(years):

        if year != earliest_year: # earliest year in dataset
            neigh_indices = all_indices[years==year-1]
            this_neigh_mask = np.zeros((pad_len))
            this_neigh_mask[len(neigh_indices):]=-np.inf
            neigh_feat = trans_features[neigh_indices]
            neigh_feat = np.vstack([neigh_feat,np.zeros((pad_len-len(neigh_feat),neigh_feat.shape[-1]))])
        else:
            this_neigh_mask = np.zeros((pad_len))
            neigh_feat = np.zeros((pad_len,trans_features.shape[-1]))

        all_neighbor_mask.append(this_neigh_mask)
        all_neighbor_features.append(neigh_feat)

        if year > (earliest_year+1):
            last_neigh_indices = all_indices[years==year-2]
            last_neigh_mask = np.zeros((pad_len))
            last_neigh_mask[len(last_neigh_indices):]=-np.inf
            last_neigh_feat = trans_features[last_neigh_indices]
            last_neigh_feat = np.vstack([last_neigh_feat,np.zeros((pad_len-len(last_neigh_feat),last_neigh_feat.shape[-1]))])
        else:
            last_neigh_mask = np.zeros((pad_len))
            last_neigh_feat = np.zeros((pad_len,neigh_feat.shape[-1]))

        all_last_neighbor_mask.append(last_neigh_mask)
        all_last_neighbor_features.append(last_neigh_feat)

    all_neighbor_features = np.stack(all_neighbor_features)
    all_last_neighbor_features = np.stack(all_last_neighbor_features)
    all_neighbor_mask = np.expand_dims(np.stack(all_neighbor_mask),axis=-1)
    all_last_neighbor_mask = np.expand_dims(np.stack(all_last_neighbor_mask),axis=-1)

    return all_neighbor_features,all_neighbor_mask, all_last_neighbor_features,all_last_neighbor_mask


def get_train_test_data(mb_y,features,all_neighbor_features,all_last_neighbor_features,sites,all_neighbor_mask,all_last_neighbor_mask,years,test_year,trans_years):
    """ Returns the training and test dataset splits
    args:
        test_year (int): specifies which year is for testing
    """
    train_mb = mb_y[years!=test_year]
    train_features = features[years!=test_year]
    train_neigh_features = all_neighbor_features[years!=test_year]
    train_last_neigh_features = all_last_neighbor_features[years!=test_year]
    train_sites = sites[years!=test_year]
    train_neigh_mask = all_neighbor_mask[years!=test_year]
    train_last_neigh_mask = all_last_neighbor_mask[years!=test_year]
    train_trans_year = trans_years[years!=test_year]
    train_data = (train_mb,train_features,train_neigh_features,train_last_neigh_features,train_sites,train_neigh_mask,train_last_neigh_mask,train_trans_year)

    test_mb = mb_y[years==test_year]
    test_features = features[years==test_year]
    test_neigh_features = all_neighbor_features[years==test_year]
    test_last_neigh_features = all_last_neighbor_features[years==test_year]
    test_sites = sites[years==test_year]
    test_neigh_mask = all_neighbor_mask[years==test_year]
    test_last_neigh_mask = all_last_neighbor_mask[years==test_year]
    test_trans_year = trans_years[years==test_year]
    test_data = (test_mb,test_features,test_neigh_features,test_last_neigh_features,test_sites,test_neigh_mask,test_last_neigh_mask,test_trans_year)

    return train_data,test_data


def gather_data(test_year,season,pad_len,data_dir="../data/",model_type="standard"):
    """ Loads and processes data for this test year & season
    args:
        test_year (int): specifies the year being used as the test year
        season (str): winter/summer, determines which labels are loaded
        pad_len (int): determines the amount of padding for the site neighborhood
        model_type (str): indicates whether to generate features for the standard model or time-series model
    """
    features,trans_features,mb_y,trans_years,sites,years = load_data(test_year,season,data_dir)
    if model_type=="standard":
        all_neighbor_features,all_neighbor_mask, all_last_neighbor_features,all_last_neighbor_mask = generate_traditional_neighborhood_features(trans_features,sites,years,pad_len)
    else:
        all_neighbor_features,all_neighbor_mask, all_last_neighbor_features,all_last_neighbor_mask = generate_time_series_neighborhood_features(trans_features,sites,years,pad_len)
    train_data,test_data = get_train_test_data(mb_y,features,all_neighbor_features,all_last_neighbor_features,sites,all_neighbor_mask,all_last_neighbor_mask,years,test_year,trans_years)

    return train_data,test_data,years



def load_data_inference(season,data_dir="../data/"):
    """ Loads the data required for Inference.
    """
    if season == 'summer':
        mb_y = np.load(data_dir+"mb_s.npy")
    elif season == 'winter':
        mb_y = np.load(data_dir+"mb_w.npy")

    features = np.load(data_dir+"features.npy") # x,y,z,t
    years = features[:,-1]

    sc = StandardScaler()
    trans_years = sc.fit_transform(np.expand_dims(features[:,-1],axis=-1))
    trans_x = sc.fit_transform(features[:,:-1])

    y_sc = StandardScaler()
    trans_y = y_sc.fit_transform(np.expand_dims(mb_y,axis=-1))
    trans_features = np.hstack([trans_x,trans_y])

    return trans_features,trans_years,years,sc


def generate_inference_neighborhood_features(trans_features,trans_years,years,test_year,pad_len,model_type,batch_size=100):
    """ Generates neighborhood features for inference.
    args:
        test_year (int): specifies the year being used as the test year
        model_type (str): indicates whether to generate features for the standard model or time-series model
        batch_size (int): copies the neighborhood features batch_size number of times
    """
    if model_type == "standard":
        first_test_year = test_year
    else:
        first_test_year = test_year-1

    all_indices = np.array([i for i in range(len(trans_features))])

    neigh_indices = all_indices[years==first_test_year]
    this_neigh_mask = np.zeros((pad_len))
    this_neigh_mask[len(neigh_indices):]=-np.inf
    neigh_feat = trans_features[neigh_indices]
    neigh_feat = np.vstack([neigh_feat,np.zeros((pad_len-len(neigh_feat),neigh_feat.shape[-1]))])

    last_neigh_indices = all_indices[years==first_test_year-1]
    last_neigh_mask = np.zeros((pad_len))
    last_neigh_mask[len(last_neigh_indices):]=-np.inf
    last_neigh_feat = trans_features[last_neigh_indices]
    last_neigh_feat = np.vstack([last_neigh_feat,np.zeros((pad_len-len(last_neigh_feat),last_neigh_feat.shape[-1]))])

    trans_year = trans_years[years==test_year][0]
    batch_trans_year = np.stack([trans_year for _ in range(batch_size)])

    batch_neigh_x = np.stack([neigh_feat for _ in range(batch_size)])
    batch_mask = np.stack([this_neigh_mask for _ in range(batch_size)])
    batch_last_neigh_x = np.stack([last_neigh_feat for _ in range(batch_size)])
    batch_last_mask = np.stack([last_neigh_mask for _ in range(batch_size)])

    return batch_neigh_x,batch_mask, batch_last_neigh_x,batch_last_mask, batch_trans_year


def gather_data_inference(test_year,season,pad_len,model_type,data_dir="../data/",batch_size=100):
    """ Loads and processes data for inference.
    """
    trans_features,trans_years,years,sc = load_data_inference(season,data_dir)
    batch_neigh_x,batch_mask, batch_last_neigh_x,batch_last_mask, batch_trans_year = generate_inference_neighborhood_features(trans_features,trans_years,years,test_year,pad_len,model_type,batch_size=batch_size)
    inference_data = (sc, batch_neigh_x,batch_mask, batch_last_neigh_x,batch_last_mask, batch_trans_year)

    return inference_data


def get_glacier_data(sc,data_dir="../data/"):
    """ Processes the data for entire glacier.
    """
    mb_variance_matrix = np.zeros((683,701)).astype("float32")
    mass_balance_matrix = np.zeros((683,701)).astype("float32")
    g_indices = [] # to feed to model
    g_features = []

    with open(data_dir+"LaJokull-2015-maiWV100x100.dat") as infile:
        lines = infile.readlines()

        for line in lines:
            x,y,elev = line.split(" ")
            x,y,elev = float(x),float(y),float(elev)
            x_coord,y_coord = int((x-408200)/100),int((y-430400)/100)
            
            if elev == 1.70141e+38:
                mass_balance_matrix[x_coord,y_coord]=np.nan
                mb_variance_matrix[x_coord,y_coord]=np.nan
            else:
                g_indices.append((x_coord,y_coord))
                g_features.append(sc.transform(np.expand_dims(np.array([x,y,elev]),axis=0)))
                
    g_features = np.vstack(g_features)
    return g_features,g_indices,mass_balance_matrix,mb_variance_matrix

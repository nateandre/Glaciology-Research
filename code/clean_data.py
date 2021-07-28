""" Used to clean the glaciology (mass balance) data.

"""

import pandas as pd
import numpy as np
import os


def main():
    clean_all_data()


def clean_all_data(data_dir="../data/"):
    """ Used to clean the (x,y,z,t) values for all sites, for all years
    """
    features = [] # x,y,z,t
    mb_s = []
    mb_w = []
    sites = []

    for data_file in os.listdir(data_dir+"mb_by_year/"):
        year = int((data_file.split(".")[0]).split("-")[-1])
        dframe = pd.read_csv(data_dir+"mb_by_year/"+data_file)
        dframe = dframe[dframe['lat']>0] # removes invalid rows (no observation for this site is available)

        this_site = dframe['site'].to_numpy()
        x_coord = dframe['x-i93'].to_numpy()
        y_coord = dframe['y-i93'].to_numpy()
        z_coord = dframe['elvev m.a.s.l.'].to_numpy()
        t_coord = np.array([year for _ in range(len(z_coord))])
        this_features = np.swapaxes(np.vstack([x_coord,y_coord,z_coord,t_coord]),0,1)
        this_mb_s = dframe['bs'].to_numpy()
        this_mb_w = dframe['bw'].to_numpy()
        
        features.append(this_features)
        mb_s.append(this_mb_s)
        mb_w.append(this_mb_w)
        sites.append(this_site)
        
    features = np.vstack(features)
    mb_s = np.concatenate(mb_s)
    mb_w = np.concatenate(mb_w)
    sites = np.concatenate(sites)
    print(features.shape,mb_s.shape,mb_w.shape,sites.shape)

    np.save(data_dir+"mb_s.npy",mb_s)
    np.save(data_dir+"mb_w.npy",mb_w)
    np.save(data_dir+"features.npy",features)
    np.save(data_dir+"sites.npy",sites)


if __name__=="__main__":
    main()

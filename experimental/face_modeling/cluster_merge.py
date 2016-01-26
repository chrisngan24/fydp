"""
Creates training data from semi-supervised technique.
Give the script a set of data that has a specific turn.
It then clusters the events. Each cluster represents a component of the
timed event (ie. starting cluster of events, middle nad end of events)
"""
import datetime
import os
import sys
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json
import matplotlib.pyplot as plt


import head_features
from visualize import fit_pca, plot_diagnostics


def relabel_by_time(y):
    """
    Remaps clusters so they appear in time order
    """
    mapper = {}
    i = 0
    out = []
    for x in y:
        if not mapper.has_key(x):
            mapper[x] = i
            i+= 1
        out.append(mapper[x])
    return np.array(out)
            
def get_active_features(df, ignore_columns=[]):
    active_columns = df.columns.values.tolist()
    for c in ignore_columns:
        if co in cols:
            active_columns.remove(co)
    return active_columns


def cluster_training_signals(df, active_features, k):
    """
    Cluster in a lower dimension
    """
    pca = PCA(n_components=2)
    kmean = KMeans(n_clusters=k)
    # X = pca.fit_transform(df[active_features])
    X = df[active_features]
    Y = kmean.fit_predict(X)
    Y = relabel_by_time(Y.tolist())
    return Y

         

def generate_training_set(director, k=4, window_size=10,relevant_features=[]):
    """
    Given the directory of data files,
    cluster the event points and saves the results
    in the merged folder. It appends an extra `class` column
    to represente the labeling results.
    """
    training_data = pd.DataFrame()
    active_features = []
    for csv in os.listdir(director):
        if not csv.find('.csv') == -1:
            fi_path = '%s/%s' % (director, csv)
            df = pd.read_csv(fi_path)
            print fi_path
            # Save to raw so the original data is kept. We are interested in 
            # keeping the original data for these rows
            df['noseX_raw'] = df['noseX']
            df['noseY_raw'] = df['noseY']
            # features
            df, active_features = head_features.apply_feature_engineering(df, relevant_features)
            df.fillna(0,inplace=True)
            df_w, active_features = head_features.generate_windows(df, 
                window = window_size,
                relevant_features=active_features,
                )
            training_data = pd.concat(
                [training_data, df_w.loc[0:(len(df_w)-window_size)]]
                )
    df_w = training_data
    print 'Now clustering the data'
    # active_columns = get_active_features(df, ignore_columns)

    Y = cluster_training_signals(
        df_w, 
        active_features, 
        k,
        )
    df_w['class'] = Y
    print "Number of data points clustered:", len(df_w)
    print "Features used to cluster:\n"
    for c in active_features:
        print "\t%s" % c
    return df_w, active_features


  
            

if __name__ == '__main__':
    fi = open('config.json', 'r')
    config = json.loads(reduce(lambda x, y: x + y, fi.readlines()))
    window_size = config['window_size']
    relevant_features = config['relevant_features']

    # the directory (labeled data) we are looking to load head turns from
    data_dir = sys.argv[1]
    # number of clusters to cluster the data on
    k = int(sys.argv[2])
    base_dir = 'data' # dir to read data from
    m_dir = '%s/%s' % (base_dir, data_dir)
    output_dir = 'data/merged'
    ignore_columns = ['date','frameIndex', 'class', 'time', 'noseX_raw', 'noseY_raw']
    df, active_features = generate_training_set(m_dir, k=k, window_size=window_size,
            relevant_features=relevant_features)
    df.to_csv('%s/%s.csv' % (output_dir, data_dir), index=False)
    plot_diagnostics(df, active_features, '%s/%s' % (output_dir, data_dir))
    config['active_features'] = active_features
    with open('config.json', 'w') as outfile:
        json.dump(config, outfile)


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


import features


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
            
def fit_pca(df, active_features, k = 2):
    pca = PCA(n_components=k)
    X = pca.fit_transform(df[active_features])
    return X

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

         

def generate_windows(df, window=10, relevant_features = []):
    """
    Take the future points - up to a specific window size -
    and add it to the current row as a set of features
    """
    points = []
    cols = df.columns.values.tolist()   
    active_features = set()
    for i, r in df.iterrows():
        w_start = i
        w_end   = min(i + 100, len(df)-1)
        row = r.to_dict()
        # drop the tail end of columns
        df_w = df.loc[w_start:w_end].reset_index(drop=True)
        for j in xrange(0,window):
            if j < len(df_w):
                window_row = df_w.loc[j].to_dict()
            else:
                window_row = None
            for c in cols:
                if c in relevant_features:
                    name = '%s_%s' % (c, j)
                    row[name] = window_row[c] if window_row != None else None
                    if not name in active_features:
                        active_features.add(name)
        points.append(row)

    return pd.DataFrame(points), list(active_features)


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
            df, active_features = features.apply_feature_engineering(df, relevant_features)
            df.fillna(0,inplace=True)
            df_w, active_features = generate_windows(df, 
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


def plot_diagnostics(df, active_features, output_dir):
    df['date'] = df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x).date())
    total_days = len(df.groupby('date'))
    i=1
    plt.figure(figsize=(20,10))
    X = fit_pca(df, active_features)
    for date, df_g in df.groupby('date'):
        plt.subplot(total_days, 1, i)
        plt.scatter(list(xrange(len(df_g))), df_g['noseX_raw'], c=df_g['class'])
        plt.title(date)
        i += 1
    print 'Saving plots to :', output_dir
    plt.savefig('%s-plots.png' % (output_dir))
   
            

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


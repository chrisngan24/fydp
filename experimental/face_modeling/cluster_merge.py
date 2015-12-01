import os
import sys
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json


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
            
def cluster_training_signals(df, active_features, k):
    pca = PCA(n_components=2)
    kmean = KMeans(n_clusters=k)
    X = pca.fit_transform(df[active_features])
    Y = kmean.fit_predict(X)
    Y = relabel_by_time(Y.tolist())
    return Y

         

def generate_windows(df, window=10, ignore_columns = []):
    """
    Apply the future windows to the dataframe
    """
    points = []
    cols = df.columns.values.tolist()
    for ic in ignore_columns:
        cols.remove(ic)
    for i, r in df.iterrows():
        w_start = i
        w_end   = min(i + 100, len(df)-1)
        row = r.to_dict()
        df_w = df.loc[w_start:w_end].reset_index(drop=True)
        for j in xrange(0,window):
            if j < len(df_w):
                window_row = df_w.loc[j].to_dict()
            else:
                window_row = None
            for c in cols:
                name = '%s_%s' % (c, j)
                row[name] = window_row[c] if window_row != None else None
        points.append(row)

    return pd.DataFrame(points)


def generate_training_set(director, k=4, window_size=10,ignore_columns = []):
    """
    """
    training_data = pd.DataFrame()
    active_columns = []
    for csv in os.listdir(director):
        if not csv.find('.csv') == -1:
            fi_path = '%s/%s' % (director, csv)
            df = pd.read_csv(fi_path)
            df['noseX_raw'] = df['noseX']
            df['noseY_raw'] = df['noseY']
            df = features.apply_feature_engineering(df, ignore_columns).fillna(0)
            df_w = generate_windows(df, 
                window = window_size,
                ignore_columns=ignore_columns,
                )
            training_data = pd.concat(
                [training_data, df_w.loc[0:(len(df_w)-window_size)]]
                )

    df_w = training_data
    active_columns = df.columns.values.tolist()
    for c in ignore_columns:
        active_columns.remove(c)

    Y = cluster_training_signals(
        df_w, 
        active_columns, 
        k,
        )
    df_w['class'] = Y
    print len(df_w)
    return df_w 


            

if __name__ == '__main__':
    fi = open('config.json', 'r')
    config = json.loads(reduce(lambda x, y: x + y, fi.readlines()))
    window_size = config['window_size']

    data_dir = sys.argv[1]
    k = int(sys.argv[2])
    base_dir = 'data'
    m_dir = '%s/%s' % (base_dir, data_dir)
    output_dir = 'data/merged'
    df = generate_training_set(m_dir, k=k, window_size=window_size,
            ignore_columns=['time', 'noseX_raw', 'noseY_raw'])
    df.to_csv('%s/%s.csv' % (output_dir, data_dir), index=False)


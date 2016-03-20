import os
import json 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.ensemble import  RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC 

from head_signal_sentiment_features import compute_signal_features

import pandas as pd
import sys
import numpy as np

def load_training_data():
    data_dir = 'data/'

    direcs = map(lambda x: data_dir + x,[
        'look_left_good/events',
        'look_left_bad/events',
        'look_right_good/events',
        'look_right_bad/events',
        ])
    # flatten all event files into a single list of file paths
    event_files = reduce(lambda l,r:l + r,
            map(lambda x: 
                map(lambda y: x + '/' + y, os.listdir(x)), 
                direcs
                )
            )
    events = []
    for f in event_files:
        print 'Extracting features from:', f
        df = pd.read_csv(f)
        row = compute_signal_features(df)
        events.append(row)
    
    return pd.DataFrame(events).fillna(0)




if __name__ == '__main__':
    model_name = sys.argv[1]
    y_class = 'good_turn'
    print 'Merging data Files...'
    '''
    fi = open('config.json', 'r')
    config = json.loads(reduce(lambda x, y: x + y, fi.readlines()))
    active_features = config['active_features']

    '''
    df_cat = load_training_data()
    active_features = df_cat.columns.tolist()
    if y_class in active_features:
        active_features.remove(y_class)
    print 'Data length:', str(len(df_cat))
    svm = SVC()
    rf = RandomForestClassifier()
    
    print 'Training the file...'
    svm.fit(df_cat[active_features], df_cat[y_class])
    rf.fit(df_cat[active_features], df_cat[y_class])
    base_dir = 'models/head_sentiment/%s' % model_name
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    path = '%s/%s.pkl' % (base_dir, model_name)
    df_cols = pd.DataFrame(dict(columns=active_features))
    joblib.dump(rf, path)
    df_cols.to_csv('%s/active_features.csv' % base_dir, index=False)

    ######
    # test the model
    #####

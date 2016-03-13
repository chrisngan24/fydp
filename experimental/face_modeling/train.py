import os
import json 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.ensemble import  RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC 

import pandas as pd
import sys
import numpy as np

from visualize import fit_pca, plot_diagnostics
def load_training_data():
    df_left = pd.read_csv('data/merged/look_left.csv')
    df_right = pd.read_csv('data/merged/look_right.csv')
    df_straight = pd.read_csv('data/merged/look_straight.csv')
    ### hack
    # Assume class 0 is always when face is forward for any data set
    # Classes I expect:
    #  0 - nothing interesting with the face
    #  1 - face is starting a left rotation
    #  2 - face is ending a left rotation
    #  3 - face is starting a right rotation
    #  4 - face is ending a right rotation
    print 'Hacking classes...'
    df_right['class'] = df_right['class'].apply(lambda x: x + 2 if x > 0 else 0)
    df_cat = pd.concat([df_left, df_right, df_straight])
    return df_cat

def load_test_data():
    df_left = pd.read_csv('data/merged/look_left_test.csv')
    df_right = pd.read_csv('data/merged/look_right_test.csv')
    df_null_test = pd.read_csv('data/merged/look_straight_test.csv')
    ### hack
    # Assume class 0 is always when face is forward for any data set
    # Classes I expect:
    #  0 - nothing interesting with the face
    #  1 - face is starting a left rotation
    #  2 - face is ending a left rotation
    #  3 - face is starting a right rotation
    #  4 - face is ending a right rotation
    print 'Hacking classes...'
    df_right['class'] = df_right['class'].apply(lambda x: x + 2 if x > 0 else 0)
    df_cat = pd.concat([df_left, df_right, df_null_test])
    return df_cat



if __name__ == '__main__':
    model_name = sys.argv[1]
    print 'Merging data Files...'
    ignore_cols =['time', 'noseX_raw', 'class', 'noseY_raw']
    fi = open('config.json', 'r')
    config = json.loads(reduce(lambda x, y: x + y, fi.readlines()))
    active_cols = config['active_features']

    df_cat = load_training_data()
    print 'Data length:', str(len(df_cat))
    knn = KNeighborsClassifier(n_neighbors=10)
    rf = RandomForestClassifier()
    
    print 'Training the file...'
    knn.fit(df_cat[active_cols], df_cat['class'])
    rf.fit(df_cat[active_cols], df_cat['class'])
    base_dir = 'models/%s' % model_name
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    path = '%s/%s.pkl' % (base_dir, model_name)
    df_cols = pd.DataFrame(dict(columns=active_cols))
    #joblib.dump(knn, path)
    joblib.dump(rf, path)
    df_cols.to_csv('%s/active_features.csv' % base_dir, index=False)

    ######
    # test the model
    #####
    df_test = load_test_data()
    def print_test_data(m_df, cf, cf_string):
        Y_test = cf.predict(m_df[active_cols])
        print cf_string, ' accuracy', \
                np.sum(Y_test == m_df['class']) / float(len(m_df))
        df_test['class'] = Y_test
        plot_diagnostics(df_test, active_cols, '%s/head-turn-test-%s' % (base_dir, cf_string))

    print_test_data(df_test, knn, 'knn')
    print_test_data(df_test, rf, 'Random Forest')

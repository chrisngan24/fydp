import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import pandas as pd

def load_training_data():
    df_left = pd.read_csv('data/merged/look_left.csv')

    df_right = pd.read_csv('data/merged/look_right.csv')
    ### hack
    # Assume class 0 is always when face is forward for any data set
    print 'Hacking classes...'
    df_right['class'] = df_right['class'].apply(lambda x: x + 2 if x > 0 else 0)
    df_cat = pd.concat([df_left, df_right])
    return df_cat

if __name__ == '__main__':
    print 'Merging data Files...'
    ignore_cols =['time', 'noseX_raw', 'class', 'noseY_raw']
    df_cat = load_training_data()
    print 'Data length:', str(len(df_cat))
    knn = KNeighborsClassifier(n_neighbors=10)
    active_cols = df_cat.columns.values.tolist()
    
    for c in ignore_cols:
        active_cols.remove(c)
    print 'Training the file...'
    knn.fit(df_cat[active_cols], df_cat['class'])
    base_dir = 'models/head_turns'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    path = '%s/head_turns.pkl' % base_dir
    df_cols = pd.DataFrame(dict(columns=active_cols))
    joblib.dump(knn, path)
    df_cols.to_csv('%s/active_features.csv' % base_dir)

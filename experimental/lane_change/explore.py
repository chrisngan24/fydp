
import pandas as pd
import numpy as np
import datetime
import math
import os

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec

from ggplot import *

import json

from IPython.display import Image

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import sklearn.cluster as cluster
import sklearn.cross_validation as cross_validation
import sys

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.externals import joblib

from scipy import signal

data_direc = os.path.join( "data")
plot_direc = os.path.join("plots")
model_direc = os.path.join("models")
lane_change_models_direc = os.path.join(model_direc, "lane_changes")

ignore_columns = ["Unnamed: 0", "az", "gx", "gz_raw", "gy","ax", "ay", "theta", "time_diff", "faceBottom", "faceLeft", "faceRight", "faceTop", "isFrontFace", "noseX", "noseY", "time", "timestamp_y", "frameIndex", "timestamp_x"]
# relevant_columns = ['gz', 'gz_0', 'gz_1', 'gz_2', 'gz_3', 'gz_4', 'gz_5', 'gz_6', 'gz_7', 'gz_8', 'gz_9']
relevant_columns = ['gz', 'gz_4']

window_size=5
step=10
n_clusters=3
moving_average_size = 20

scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

def normalize(arr):
    min_x = min(arr)
    range_x = max(arr) - min_x
    return [ float(x-min_x) / float(range_x) for x in arr ]

def subtract_from_prev_val(df, col, step=1):
    """
    Subtract column value from the previous
    column value n steps away
    """
    return (df[col] - df.shift(periods=step)[col])
def generate_features(df, suffix = '_diff_', step=1, relevant_features=[], ignore_columns=[]):
    """
    Generate the features, returns a new data frame of all 
    transformed features (same length as input)
    :param df: - input data frame
    :param suffix: - the ending of the new column, default is change nothing
                     to column name
    :param step: - delta from how many index periods away
    :param ignore_columns: - what are the columns to ignore
    """
    # cols = self.get_active_columns(df, ignore_columns)
    cols = relevant_features
    deltas = {}
    for c in cols:
        deltas['%s%s'% (c, suffix)] = subtract_from_prev_val(df, c, step=step)
    df_new = pd.DataFrame(deltas)
    return df_new

def generate_windowed_df(df):
    windowed = generate_features(df,relevant_features=relevant_columns, step=step, ignore_columns=ignore_columns)
    windowed = windowed.fillna(0)
    for c in relevant_columns:
        windowed[c] = signal.detrend(df[c])
    return windowed

def generate_windows(df, window=10, ignore_columns=ignore_columns):
    """
    Apply the future windows to the dataframe
    """
    points = []
    cols = df.columns.values.tolist()
    for ic in ignore_columns:
        if ic in cols:
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
    return pd.DataFrame(points).fillna(0)

def cluster_using_kmeans(df, filename, n_components=2, n_clusters=3):
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(df)
    kmean = KMeans(n_clusters=n_clusters)
    Y = kmean.fit_predict(df)
    return Y

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, "same")

def generate_training_and_test_data(df, cluster_labels, train_percentage):
    le = preprocessing.LabelEncoder()

    df.Labels = le.fit(cluster_labels).transform(cluster_labels)  
    y = df.Labels
    X = df

    test_index = int(len(df) * train_percentage)

    X_train = X[:test_index]
    y_train = y[:test_index]             

    X_test = X[test_index:]   
    y_test = y[test_index:]

    return X_train, y_train, X_test, y_test

def random_forest(x_train, y_train, x_test, y_test): 
    clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    joblib.dump(clf, "%s/random_forest.pkl" %model_direc)
    return accuracy

def knn(x_train, y_train, x_test, y_test): 
    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    joblib.dump(clf, "%s/knn.pkl" %model_direc)
    return accuracy

def svm(x_train, y_train, x_test, y_test): 
    clf = SVC()
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    joblib.dump(clf, "%s/svm.pkl" %model_direc)
    return accuracy

def logistic_regression(x_train, y_train, x_test, y_test): 
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    joblib.dump(clf, "%s/logistic_regression.pkl" %model_direc)
    return accuracy

def train_all_models(x_train, y_train, x_test, y_test):
    print "Random forest: ", random_forest(x_train, y_train, x_test, y_test)
    print "KNN: ", knn(x_train, y_train, x_test, y_test)
    # print "SVM: ", svm(x_train, y_train, x_test, y_test)
    # print "Logistic Regression: ", logistic_regression(x_train, y_train, x_test, y_test)

def get_data(filename):
    df = pd.read_csv(filename)
    df.fillna(0, inplace=True)
    return df

def train():
    left_dfs = []
    right_dfs = []
    neg_dfs = []

    for subdir, dirs, files in os.walk(data_direc):
        for d in dirs:
            if d.startswith("left_10") and not d.startswith("left_turn"):
                df = pd.read_csv("%s/fused.csv" %os.path.join(data_direc, d))
                df['gz'] = movingaverage(df['gz'], moving_average_size)
                # df['gz'] = scaler.fit_transform(df['gz'])
                left_dfs.append(df)
            elif d.startswith("right_10") and not d.startswith("right_turn"):
                df = pd.read_csv("%s/fused.csv" %os.path.join(data_direc, d))
                df['gz'] = movingaverage(df['gz'], moving_average_size)
                # df['gz'] = scaler.fit_transform(df['gz'])
                right_dfs.append(df)
            elif d.startswith("neg_") or d.startswith("right_turn") or d.startswith("left_turn"):
                df = pd.read_csv("%s/fused.csv" %os.path.join(data_direc, d))
                df['gz'] = movingaverage(df['gz'], moving_average_size)
                # df['gz'] = scaler.fit_transform(df['gz'])
                neg_dfs.append(df)

    df_left = pd.concat(left_dfs, axis=0, join="outer", join_axes=None, ignore_index=True, keys=None, levels=None, names=None, verify_integrity=False)
    df_right = pd.concat(right_dfs, axis=0, join="outer", join_axes=None, ignore_index=True, keys=None, levels=None, names=None, verify_integrity=False)
    df_neg = pd.concat(neg_dfs, axis=0, join="outer", join_axes=None, ignore_index=True, keys=None, levels=None, names=None, verify_integrity=False)

    windowed_df_left = generate_windows(df_left, window=window_size)
    windowed_df_right = generate_windows(df_right, window=window_size)
    windowed_df_neg = generate_windows(df_neg, window=window_size)



    # left_clusters = cluster_using_kmeans(windowed_df_left, "", n_clusters=n_clusters)
    # right_clusters = cluster_using_kme    ans(windowed_df_right, "", n_clusters=n_clusters)

    # c1_left = left_clusters[np.where(left_clusters!=left_clusters[0])[0][0]]
    # c1_right = right_clusters[np.where(right_clusters!=right_clusters[0])[0][0]]
    # left_clusters = np.array(map(lambda x: 0 if x == left_clusters[0] else 2 if x == c1_left else 1, left_clusters))
    # right_clusters = np.array(map(lambda x: 0 if x == right_clusters[0] else 2 if x == c1_right else 1, right_clusters))
    # neg_clusters = np.array([left_clusters[0]]*len(windowed_df_neg))

    df_train = pd.concat([windowed_df_left, windowed_df_right], join="outer", ignore_index=True)
    
    df_train = df_train[relevant_columns]
    
    cluster_labels = cluster_using_kmeans(df_train, "", n_clusters=n_clusters)

    x_train, y_train, x_test, y_test = generate_training_and_test_data(df_train, np.array(cluster_labels), 0.99)
    train_all_models(x_train, y_train, x_test, y_test)
    print x_train.columns

def predict(direc):
    df = get_data("%s/fused.csv" %direc)
    df['gz'] = movingaverage(df['gz'], moving_average_size)
    # df['gz'] = scaler.fit_transform(df['gz'])
    windowed_df_test = generate_windows(df, window=window_size)
    windowed_df_test = windowed_df_test[relevant_columns]

    model_name = "knn"
    clf = joblib.load("%s/%s.pkl" %(model_direc, model_name))
    predicted_labels_test = clf.predict(windowed_df_test)

    windowed_df_test['theta'] = df['theta']

    plt.figure()
    plt.scatter(df.index, df["theta"], c=predicted_labels_test, cmap='jet')
    plt.title("Test Windowed Kmeans Clustering Angle (K = %s)" %str(n_clusters))
    plt.savefig("%s/%s_%s.png" %(direc, "windowed_kmeans_test_theta", str(n_clusters)))

    plt.figure()
    plt.scatter(df.index, df["gz"], c=predicted_labels_test, cmap='jet')
    plt.title("Test Windowed Kmeans Clustering Angular Velocity (K = %s) " %str(n_clusters))
    plt.savefig("%s/%s_%s.png" %(direc, "windowed_kmeans_test_gz", str(n_clusters)))

    with open("%s/events.txt" %direc, 'w') as f:
        f.write(str(detect_events(df, predicted_labels_test)))

def detect_events(df, predicted_labels_test):
    null_label = predicted_labels_test[0]

    state = 0

    events = {
            "left_lc_start": set(),
            "left_lc_end"  : set(),
            "right_lc_start": set(),
            "right_lc_end"  : set(),
            }

    left_lc_start = 0
    right_lc_start = 0
    left_lc_end = 0
    right_lc_end = 0


    pos_label = 2
    neg_label = 1
    null_label = 0

    left_lane_sequence = [null_label, pos_label, neg_label, pos_label, null_label]
    right_lane_sequence = [null_label, neg_label, pos_label, neg_label, null_label]

    left_index = 0
    right_index = 0

    for i in xrange(len(predicted_labels_test)):
        if predicted_labels_test[i] == left_lane_sequence[left_index]:
            left_index = (left_index + 1) % len(left_lane_sequence)
            print 'left', i, left_index, left_lane_sequence[left_index]

        if predicted_labels_test[i] == right_lane_sequence[right_index]:
            right_index = (right_index + 1) % len(right_lane_sequence)
            print 'right', i, right_index, right_lane_sequence[right_index]

        if left_index == 1:
            left_lc_start = i

        if right_index == 1:
            right_lc_start = i

        if left_index == len(left_lane_sequence) - 1:
            left_index = 0
            right_index = 0
            left_lc_end = i

        elif right_index == len(right_lane_sequence) - 1:
            left_index = 0
            right_index = 0
            right_lc_end = i

        if left_lc_start > 0 and left_lc_end > 0 and left_lc_end - left_lc_start > 30:
            events["left_lc_start"].add(left_lc_start)
            events["left_lc_end"].add(left_lc_end)

        if right_lc_start > 0 and right_lc_end > 0 and right_lc_end - right_lc_start > 30:
            events["right_lc_start"].add(right_lc_start)
            events["right_lc_end"].add(right_lc_end)

    for k, v in events.iteritems():
        events[k] = sorted(list(v))

    events_indices = []
    for i in xrange(len(events['left_lc_start'])):
        t = (events['left_lc_start'][i], events['left_lc_end'][i], 'left_lane_change')
        events_indices.append(t)

    for i in xrange(len(events['right_lc_start'])):
        t = (events['right_lc_start'][i], events['right_lc_end'][i], 'right_lane_change')
        events_indices.append(t)

    return events_indices

if __name__ == "__main__":
    # train()
    
    # left_dfs = []
    # right_dfs = []
    # neg_dfs = []

    # for subdir, dirs, files in os.walk(data_direc):
    #     for d in dirs:
    #         predict(os.path.join(data_direc, d))

    predict(os.path.join(data_direc))




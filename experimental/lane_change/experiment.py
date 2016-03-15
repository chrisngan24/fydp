import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import datetime
from sklearn.externals import joblib
import fastdtw
import numpy as np

from collections import Counter

data_direc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
plot_direc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "plots")
model_direc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
lane_change_models_direc = os.path.join(model_direc, "lane_changes")

ignore_columns = ["Unnamed: 0", "time_diff", "faceBottom", "faceLeft", "faceRight", "faceTop", "isFrontFace", "noseX", "noseY", "time", "timestamp_y", "frameIndex", "timestamp_x"]
active_columns = ["ax", "ax_0", "ax_1", "ax_2", "ax_3", "ax_4", "ax_5", "ax_6",
       "ax_7", "ax_8", "ax_9", "ay", "ay_0", "ay_1", "ay_2", "ay_3",
       "ay_4", "ay_5", "ay_6", "ay_7", "ay_8", "ay_9", "az", "az_0",
       "az_1", "az_2", "az_3", "az_4", "az_5", "az_6", "az_7", "az_8",
       "az_9", "gx", "gx_0", "gx_1", "gx_2", "gx_3", "gx_4",
       "gx_5", "gx_6", "gx_7", "gx_8", "gx_9", "gy", "gy_0", "gy_1",
       "gy_2", "gy_3", "gy_4", "gy_5", "gy_6", "gy_7", "gy_8", "gy_9",
       "gz", "gz_0", "gz_1", "gz_2", "gz_3", "gz_4", "gz_5", "gz_6",
       "gz_7", "gz_8", "gz_9", "theta", "theta_0", "theta_1", "theta_2", "theta_3",
       "theta_4", "theta_5", "theta_6", "theta_7", "theta_8", "theta_9"]

n_clusters = 3
window_size = 10

def generate_windows(df, window=10, ignore_columns = []):
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

    return pd.DataFrame(points)

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, "same")

def filter_features(df):
    return df[active_columns]

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

def cluster_using_kmeans(df, filename, n_components=2, n_clusters=3):
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(df)
    kmean = KMeans(n_clusters=n_clusters)
    Y = kmean.fit_predict(df)
    return Y

def get_data(filename):
    df = pd.read_csv("%s/%s.csv" %(data_direc, filename))
    df.fillna(0, inplace=True)
    # df = df.drop("Unnamed: 0", 1)
    return df

def train_all_models(x_train, y_train, x_test, y_test):
    print "Random forest: ", random_forest(x_train, y_train, x_test, y_test)
    print "KNN: ", knn(x_train, y_train, x_test, y_test)
    # print "SVM: ", svm(x_train, y_train, x_test, y_test)
    # print "Logistic Regression: ", logistic_regression(x_train, y_train, x_test, y_test)

def train(filename):
    df_train = get_data(filename)
    cluster_labels_train = cluster_using_kmeans(prepare_data(df_train, for_clustering=True), "Train")
    x_train, y_train, x_test, y_test = generate_training_and_test_data(prepare_data(df_train), cluster_labels_train, 0.8)
    train_all_models(x_train, y_train, x_test, y_test)
    return df_train

def test(filename, model_name):
    df_test = get_data(filename)
    cluster_labels_test = cluster_using_kmeans(prepare_data(df_test, for_clustering=True), "Test")
    clf = joblib.load("%s/%s.pkl" %(model_direc, model_name))
    x_df_test = prepare_data(df_test)
    predicted_labels_test = clf.predict(x_df_test)
    features = x_df_test.columns.values.tolist()
    plt.figure()
    plt.scatter(df_test["timestamp_x"], df_test["theta"], c=predicted_labels_test)
    plt.title("Testing %s Classification %s" %(model_name, ",".join(features)))
    plt.savefig("%s/test_%s_%s.png" %(plot_direc, model_name, "_".join(features)))
    return df_test

def get_annotations():

    left_indices = []
    with open("%s/left.txt" %data_direc, "r") as f:
        for line in f:
            left_indices.append(map(int, line.split()))

    right_indices = []
    with open("%s/right.txt" %data_direc, "r") as f:
        for line in f:
            right_indices.append(map(int, line.split()))

    return left_indices, right_indices

def train():
    left_dfs = []
    right_dfs = []
    neg_dfs = []

    for subdir, dirs, files in os.walk(data_direc):
        for d in dirs:
            if d.startswith("left_") and not d.startswith("left_turn"):
                left_dfs.append(pd.read_csv("%s/fused.csv" %os.path.join(data_direc, d)))
            elif d.startswith("right_") and not d.startswith("right_turn"):
                right_dfs.append(pd.read_csv("%s/fused.csv" %os.path.join(data_direc, d)))
            elif d.startswith("neg_"):
                neg_dfs.append(pd.read_csv("%s/fused.csv" %os.path.join(data_direc, d)))

    left = pd.concat(left_dfs, axis=0, join="outer", join_axes=None, ignore_index=True,
       keys=None, levels=None, names=None, verify_integrity=False)
    right = pd.concat(right_dfs, axis=0, join="outer", join_axes=None, ignore_index=True,
       keys=None, levels=None, names=None, verify_integrity=False)
    neg = pd.concat(neg_dfs, axis=0, join="outer", join_axes=None, ignore_index=True,
       keys=None, levels=None, names=None, verify_integrity=False)

    windowed_left = generate_windows(left, window=window_size, ignore_columns=ignore_columns)
    windowed_left = windowed_left.fillna(0)

    windowed_right = generate_windows(right, window=window_size, ignore_columns=ignore_columns)
    windowed_right = windowed_right.fillna(0)

    windowed_neg = generate_windows(neg, window=window_size, ignore_columns=ignore_columns)
    windowed_neg = windowed_neg.fillna(0)

    left_clusters = cluster_using_kmeans(windowed_left, "", n_clusters=n_clusters)
    right_clusters = cluster_using_kmeans(windowed_right, "", n_clusters=n_clusters)
    neg_clusters = cluster_using_kmeans(windowed_neg, "", n_clusters=1)

    plt.figure()
    plt.scatter(left.index, left["theta"], c=left_clusters)
    plt.title("Left Lane Changes Train Windowed Kmeans Clustering (K = %s)" %str(n_clusters))
    plt.savefig("%s/%s_%s.png" %(plot_direc, "windowed_kmeans_left_train", str(n_clusters)))

    plt.figure()
    plt.scatter(right.index, right["theta"], c=right_clusters)
    plt.title("Right Lane Changes Train Windowed Kmeans Clustering (K = %s)" %str(n_clusters))
    plt.savefig("%s/%s_%s.png" %(plot_direc, "windowed_kmeans_right_train", str(n_clusters)))

    plt.figure()
    plt.scatter(neg.index, neg["theta"], c=neg_clusters)
    plt.title("Negative Lane Changes Train Windowed Kmeans Clustering")
    plt.savefig("%s/%s.png" %(plot_direc, "windowed_kmeans_neg_train"))

    neg_clusters = np.array([0] * len(neg_clusters))

    c1_left = left_clusters[np.where(left_clusters!=left_clusters[0])[0][0]]

    c1_right = right_clusters[np.where(right_clusters!=right_clusters[0])[0][0]]

    left_clusters = np.array(map(lambda x: 0 if x == left_clusters[0] else 2 if x == c1_left else 1, left_clusters))
    right_clusters = np.array(map(lambda x: 0 if x == right_clusters[0] else 2 if x == c1_right else 1, right_clusters))

    cluster_labels = np.concatenate((left_clusters, neg_clusters, right_clusters), axis=0)
    import pdb; pdb.set_trace()
    
    windowed_df_train = pd.concat([windowed_left, windowed_right], axis=0, join="outer", join_axes=None, ignore_index=True,
       keys=None, levels=None, names=None, verify_integrity=False).fillna(0)
    # cluster_labels = cluster_using_kmeans(windowed_df_train, "", n_clusters=n_clusters)

    windowed_df_train = filter_features(windowed_df_train)

    x_train, y_train, x_test, y_test = generate_training_and_test_data(windowed_df_train, cluster_labels, 0.99)
    train_all_models(x_train, y_train, x_test, y_test)

if __name__ == "__main__":

    train()

    df_test = get_data("fused")

    windowed_df_test = utils.generate_windows(df_test, window=window_size, ignore_columns=ignore_columns)
    windowed_df_test = windowed_df_test.fillna(0)

    windowed_df_test = filter_features(windowed_df_test)

    model_name = "knn"
    clf = joblib.load("%s/%s.pkl" %(model_direc, model_name))
    predicted_labels_test = clf.predict(windowed_df_test)

    plt.figure()
    plt.scatter(df_test.index, df_test["theta"], c=predicted_labels_test)
    plt.title("Test Windowed Kmeans Clustering (K = %s)" %str(n_clusters))
    plt.savefig("%s/%s_%s.png" %(plot_direc, "windowed_kmeans_test", str(n_clusters)))

    null_label = predicted_labels_test[0]

    # 0: NO BUMP
    # 1: ONE POS BUMP
    # 2: ONE NEG BUMP
    # 3: WAITING FOR POS BUMP TO FINISH
    # 4: WAITING FOR NEG BUMP TO FINISH

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
    pos_label = null_label
    neg_label = null_label

    for i in xrange(len(predicted_labels_test)-5):
        if state == 0 and predicted_labels_test[i] != null_label and (predicted_labels_test[i+5] == predicted_labels_test[i]):
            if df_test["theta"][i+5] > df_test["theta"][i]:
                if pos_label == null_label:
                    pos_label = predicted_labels_test[i]
                state = 3
                left_lc_start = i
                left_lc_end = 0
            else:
                if neg_label == null_label:
                    neg_label = predicted_labels_test[i]
                state = 4
                right_lc_start = i
                right_lc_end = 0
        elif state == 3 and predicted_labels_test[i] == null_label:
            state = 5
        elif state == 4 and predicted_labels_test[i] == null_label:
            state = 6
        elif state == 5 and predicted_labels_test[i] != pos_label and predicted_labels_test[i] != null_label:
            state = 1
        elif state == 6 and predicted_labels_test[i] != neg_label and predicted_labels_test[i] != null_label:
            state = 2
        elif state == 1 and predicted_labels_test[i] == null_label:
            state = 0
            left_lc_end = i
        elif state == 2 and predicted_labels_test[i] == null_label:
            state = 0
            right_lc_end = i

        if left_lc_start > 0 and left_lc_end > 0 and left_lc_end - left_lc_start > 20:
            events["left_lc_start"].add(left_lc_start)
            events["left_lc_end"].add(left_lc_end)
        if right_lc_start > 0 and right_lc_end > 0 and right_lc_end - right_lc_start > 20:
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


    print events_indices



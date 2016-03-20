from event_annotator import EventAnnotator
import lane_features
from sklearn.externals import joblib
import os
import pandas as pd
import numpy as np
from fastdtw import fastdtw
import json
import util
from scipy import signal
from sklearn import preprocessing

from collections import OrderedDict

class LaneAnnotator(EventAnnotator):
    def __init__(self):
        m_dir = os.path.dirname(__file__)
        self.base_dir = os.path.join(m_dir, '../models/lane_changes/')
        
        self.left = []
        self.right = []
        self.left_turn = []
        self.right_turn = []
        self.neg = []

        dtw_models_direc = os.path.join(self.base_dir, 'lane_changes')

        for subdir, dirs, files in os.walk(dtw_models_direc):
            for d in dirs:
                if d.startswith("left_") and not d.startswith("left_turn"):
                    self.left.append(pd.read_csv("%s/fused.csv" %os.path.join(dtw_models_direc, d)))
                elif d.startswith("right_") and not d.startswith("right_turn"):
                    self.right.append(pd.read_csv("%s/fused.csv" %os.path.join(dtw_models_direc, d)))
                elif d.startswith("left_turn"):
                    self.left_turn.append(pd.read_csv("%s/fused.csv" %os.path.join(dtw_models_direc, d)))
                elif d.startswith("right_turn"):
                    self.right_turn.append(pd.read_csv("%s/fused.csv" %os.path.join(dtw_models_direc, d)))
                else:
                    self.neg.append(pd.read_csv("%s/fused.csv" %os.path.join(dtw_models_direc, d)))

        self.model = joblib.load('%s/knn.pkl' % self.base_dir) 
        config_fi = open('%s/config.json' % self.base_dir, 'r')
        self.config = json.loads(reduce(lambda x, y: x + y, config_fi))
        self.window_size = self.config['window_size']
        self.ignore_columns = self.config['ignore_columns']
        self.active_features = self.config['active_features']
        self.moving_average_size = self.config['moving_average_size']

        self.events = []

    def is_valid_event(self, signal, e_type):
        left_cost = min([fastdtw(signal, x['theta'].tolist())[0] for x in self.left])
        right_cost = min([fastdtw(signal, x['theta'].tolist())[0] for x in self.right])
        neg_cost = min([fastdtw(signal, x['theta'].tolist())[0] for x in self.neg])

        if e_type == 'left':
            return left_cost < right_cost and left_cost < neg_cost
        else:
            return right_cost < left_cost and right_cost < neg_cost


    def annotate_events(self, df, index_col='frameIndex'):
        df['gz'] = util.movingaverage(df['gz'], self.moving_average_size)
        windowed_df_test = util.generate_windows(df, window=self.window_size, ignore_columns=self.ignore_columns)
        windowed_df_test = windowed_df_test[self.active_features]

        predicted_labels_test = self.model.predict(windowed_df_test)
        windowed_df_test['theta'] = df['theta']

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

        l_start = OrderedDict()
        r_start = OrderedDict()

        for i in xrange(len(predicted_labels_test) - 2):
            # starts with OO
            if predicted_labels_test[i+1] == pos_label and predicted_labels_test[i+2] == pos_label \
            and (predicted_labels_test[i] == null_label or predicted_labels_test[i] == neg_label or i == 0):
                l_start[i] = 0
                for k in r_start.keys():
                    r_start[k] += 1
            # starts with <<
            if predicted_labels_test[i+1] == neg_label and predicted_labels_test[i+2] == neg_label \
            and (predicted_labels_test[i] == null_label or predicted_labels_test[i] == pos_label or i == 0):
                r_start[i] = 0
                for k in l_start.keys():
                    l_start[k] += 1
            # ends with OO
            if predicted_labels_test[i] == pos_label and predicted_labels_test[i+1] == pos_label \
            and (predicted_labels_test[i+2] == null_label or predicted_labels_test[i+2] == neg_label):
                found = False
                for k, v in l_start.items():
                    if v >= 1:
                        del l_start[k]
                        if found:
                            continue
                        signal = df.iloc[k:i]['theta'].tolist()
                        if self.is_valid_event(signal, 'left'):
                            if (len(events['right_lc_end']) > 0 and k > max(events['right_lc_end']) or len(events['right_lc_end']) == 0) \
                            and (len(events['left_lc_end']) > 0 and k > max(events['left_lc_end']) or len(events['left_lc_end']) == 0):
                                events['left_lc_start'].add(k)
                                events['left_lc_end'].add(i)
                                found = True

            # ends with <<
            if predicted_labels_test[i] == neg_label and predicted_labels_test[i+1] == neg_label \
            and (predicted_labels_test[i+2] == null_label or predicted_labels_test[i+2] == pos_label):
                found = False
                for k, v in r_start.items():
                    if v >= 1:
                        del r_start[k]
                        if found:
                            continue
                        signal = df.iloc[k:i]['theta'].tolist()
                        if self.is_valid_event(signal, 'right'):
                            if (len(events['right_lc_end']) > 0 and k > max(events['right_lc_end']) or len(events['right_lc_end']) == 0) \
                            and (len(events['left_lc_end']) > 0 and k > max(events['left_lc_end']) or len(events['left_lc_end']) == 0):
                                events['right_lc_start'].add(k)
                                events['right_lc_end'].add(i)
                                found = True

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
        
        return events, events_indices



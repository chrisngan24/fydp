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
                if d.startswith("left_lane"):
                    self.left.append(pd.read_csv("%s/fused.csv" %os.path.join(dtw_models_direc, d)))
                elif d.startswith("right_lane"):
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

    def filter_events(self, df, events):
        left_dtw_model = util.normalize(self.left[0]['theta'].tolist())
        right_dtw_model = util.normalize(self.right[0]['theta'].tolist())
        neg_dtw_model = util.normalize(self.neg[1]['theta'].tolist())
        left_turn_dtw_model = util.normalize(self.left_turn[0]['theta'].tolist())
        right_turn_dtw_model = util.normalize(self.right_turn[0]['theta'].tolist())
        
        left_indices_to_be_removed = []
        right_indices_to_be_removed = []
        
        for i in xrange(len(events['left_lc_start'])):
            signal = util.normalize(df.iloc[events['left_lc_start'][i]:events['left_lc_end'][i]+1]['theta'].tolist())
            cost = fastdtw(signal, left_dtw_model)[0]
            if cost > fastdtw(signal, neg_dtw_model)[0] or cost > fastdtw(signal, left_turn_dtw_model)[0]:
                left_indices_to_be_removed.append(i)

        for i in xrange(len(events['right_lc_start'])):
            signal = util.normalize(df.iloc[events['right_lc_start'][i]:events['right_lc_end'][i]+1]['theta'].tolist())
            cost = fastdtw(signal, right_dtw_model)[0]
            if cost > fastdtw(signal, neg_dtw_model)[0] or cost > fastdtw(signal, right_turn_dtw_model)[0]:
                right_indices_to_be_removed.append(i)

        events['left_lc_start'] = [i for j, i in enumerate(events['left_lc_start']) if j not in left_indices_to_be_removed]
        events['left_lc_end'] = [i for j, i in enumerate(events['left_lc_end']) if j not in left_indices_to_be_removed]

        events['right_lc_start'] = [i for j, i in enumerate(events['right_lc_start']) if j not in right_indices_to_be_removed]
        events['right_lc_end'] = [i for j, i in enumerate(events['right_lc_end']) if j not in right_indices_to_be_removed]

        return events

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

            if left_lc_start > 0 and left_lc_end > 0 and left_lc_end - left_lc_start > 40:
                events["left_lc_start"].add(left_lc_start)
                events["left_lc_end"].add(left_lc_end)

            if right_lc_start > 0 and right_lc_end > 0 and right_lc_end - right_lc_start > 40:
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
        
        return events, events_indices



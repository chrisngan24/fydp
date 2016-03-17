from event_annotator import EventAnnotator
import lane_features
from sklearn.externals import joblib
import os
import pandas as pd
import numpy as np
from fastdtw import fastdtw
import json
import util

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
        df_feat = util.generate_windows(df, window=self.window_size, ignore_columns=self.ignore_columns)
        
        # not sure if this does anything yet
        df_feat['theta'] = util.movingaverage(df_feat['theta'], 5)

        df_feat = df_feat.fillna(0)
        df_test = df_feat[self.active_features]

        predicted_labels_test = self.model.predict(df_test)
        
        null_label = predicted_labels_test[0]

        # 0: NO BUMP
        # 1: ONE POS BUMP
        # 2: ONE NEG BUMP
        # 3: WAITING FOR POS BUMP TO FINISH
        # 4: WAITING FOR NEG BUMP TO FINISH

        state = 0

        events = {
                'left_lc_start': set(),
                'left_lc_end'  : set(),
                'right_lc_start': set(),
                'right_lc_end'  : set(),
                }
        events_indices = []

        left_lc_start = 0
        right_lc_start = 0
        left_lc_end = 0
        right_lc_end = 0
        pos_label = null_label
        neg_label = null_label

        for i in xrange(len(predicted_labels_test)-5):
            if state == 0 and predicted_labels_test[i] != null_label and (predicted_labels_test[i+5] == predicted_labels_test[i]):
                if df_test['theta'][i+5] > df_test['theta'][i]:
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
                events['left_lc_start'].add(left_lc_start)
                events['left_lc_end'].add(left_lc_end)
            
            if right_lc_start > 0 and right_lc_end > 0 and right_lc_end - right_lc_start > 20:
                events['right_lc_start'].add(right_lc_start)
                events['right_lc_end'].add(right_lc_end)

        for k, v in events.iteritems():
            events[k] = sorted(list(v))

        events = self.filter_events(df, events)

        for i in xrange(len(events['left_lc_start'])):
            t = (events['left_lc_start'][i], events['left_lc_end'][i], 'left_lane_change')
            events_indices.append(t)

        for i in xrange(len(events['right_lc_start'])):
            t = (events['right_lc_start'][i], events['right_lc_end'][i], 'right_lane_change')
            events_indices.append(t)

        return events, events_indices



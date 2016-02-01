from event_annotator import EventAnnotator
import lane_features
from sklearn.externals import joblib
import os
import numpy as np
import dtw
import json
import util

class LaneAnnotator(EventAnnotator):
    def __init__(self):
        m_dir = os.path.dirname(__file__)
        self.base_dir = os.path.join(m_dir, '../models/lane_changes/')
        self.left_models = []
        self.right_models = []

        for f in os.listdir(self.base_dir):
            if f.startswith("left_"):
                model_file = open('%s/%s' % (self.base_dir, f)).read().split('\n')
                self.left_models.append(np.array([float(i) for i in model_file]))
            if f.startswith("right_"):
                model_file = open('%s/%s' % (self.base_dir, f)).read().split('\n')
                self.right_models.append(np.array([float(i) for i in model_file]))

        self.model = joblib.load('%s/random_forest.pkl' % self.base_dir) 
        config_fi = open('%s/config.json' % self.base_dir, 'r')
        self.config = json.loads(reduce(lambda x, y: x + y, config_fi))
        self.window_size = self.config['window_size']
        self.ignore_columns = self.config['ignore_columns']
        self.active_features = self.config['active_features']

        self.events = []

    def annotate_events(self, df, index_col='frameIndex'):
        df_feat = lane_features.prepare_data(df, ignore_columns=self.ignore_columns)
        df_feat.fillna(0, inplace=True)

        df_w = util.generate_windows(df_feat).dropna()
        df_w = df_w[self.active_features]

        predicted_labels = self.model.predict(df_w)

        df_feat = df_feat.loc[0:(len(df)-self.window_size)]
        df = df.loc[0:(len(df)-self.window_size)]
        
        null_label = predicted_labels[0]
        curr_label = null_label

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

        for i in xrange(len(predicted_labels)-5):
            if state == 0 and predicted_labels[i] != null_label and (predicted_labels[i+5] == predicted_labels[i]):
                if df_feat['theta'][i+5] > df_feat['theta'][i]:
                    state = 3
                    curr_label = predicted_labels[i]
                    left_lc_start = i
                    left_lc_end = 0
                else:
                    state = 4
                    curr_label = predicted_labels[i]
                    right_lc_start = i
                    left_lc_end = 0
            elif state == 3 and predicted_labels[i] != null_label and predicted_labels[i] != curr_label:
                state = 1
            elif state == 4 and predicted_labels[i] != null_label and predicted_labels[i] != curr_label:
                state = 2
            elif state == 1 and predicted_labels[i] == null_label:
                state = 0
                curr_label = null_label
                left_lc_end = i
            elif state == 2 and predicted_labels[i] == null_label:
                state = 0
                curr_label = null_label
                right_lc_end = i
            if left_lc_start > 0 and left_lc_end > 0 and left_lc_end - left_lc_start > 20:
                events['left_lc_start'].add(left_lc_start)
                events['left_lc_end'].add(left_lc_end)
            if right_lc_start > 0 and right_lc_end > 0 and right_lc_end - right_lc_start > 20:
                events['right_lc_start'].add(right_lc_start)
                events['right_lc_end'].add(right_lc_end)


        for k, v in events.iteritems():
            events[k] = sorted(list(v))

        for i in xrange(len(events['left_lc_start'])):
            t = (df.iloc[events['left_lc_start'][i]][index_col], df.iloc[events['left_lc_end'][i]][index_col], 'left_lane_change')
            events_indices.append(t)

        for i in xrange(len(events['right_lc_start'])):
            t = (df.iloc[events['right_lc_start'][i]][index_col], df.iloc[events['right_lc_end'][i]][index_col], 'right_lane_change')
            events_indices.append(t)

        return events, events_indices



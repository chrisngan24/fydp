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

        self.model = joblib.load('%s/knn.pkl' % self.base_dir) 
        config_fi = open('%s/config.json' % self.base_dir, 'r')
        self.config = json.loads(reduce(lambda x, y: x + y, config_fi))
        self.window_size = self.config['window_size']
        self.ignore_columns = self.config['ignore_columns']
        self.active_features = self.config['active_features']

        self.events = []

    def annotate_events(self, df, index_col='frameIndex'):

        df.fillna(0, inplace=True)
        df_feat = util.generate_windows(df, ignore_columns=self.ignore_columns)
        df_feat = df_feat.dropna()
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

        for i in xrange(len(events['left_lc_start'])):
            t = (events['left_lc_start'][i], events['left_lc_end'][i], 'left_lane_change')
            events_indices.append(t)

        for i in xrange(len(events['right_lc_start'])):
            t = (events['right_lc_start'][i], events['right_lc_end'][i], 'right_lane_change')
            events_indices.append(t)

        return events, events_indices



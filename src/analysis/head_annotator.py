from event_annotator import EventAnnotator
import os

import head_features
import pandas as pd
from util import generate_windows
from sklearn.externals import joblib
from collections import Counter
import json


class HeadAnnotator(EventAnnotator):
    def __init__(self):
        self.events = []
        m_dir = os.path.dirname(__file__)
        base_dir = os.path.join(m_dir, '../models/head_turns/')
        self.model = joblib.load('%s/head_turns.pkl' % base_dir) 
        config_fi = open('%s/config.json' % base_dir, 'r')
        self.config = json.loads(reduce(lambda x, y: x + y, config_fi))
        self.window_size = self.config['window_size']
        self.active_features = \
                pd.read_csv('%s/active_features.csv' % base_dir)['columns'].tolist()

    def annotate_events(self, df):
        window_size = self.window_size
        active_features = self.active_features
        df_feat = head_features.apply_feature_engineering(df, []).fillna(0)
        df_w = generate_windows(
            df_feat, 
            window = window_size,
            )
        # Cut off the tail end of the data (lots of null values)
        df_w = df_w.loc[0:(len(df_w)-window_size)]
        Y = self.model.predict(df_w[active_features])
        print Y
        # These are the raw events
        return self.find_true_events(df, Y.tolist())

    def find_true_events(self, df, Y):
        """
        THis is a hacky way to combine classified signal points
        to generate signals
        """
        events = {
            'left_turn_start': [],
            'left_turn_end'  : [],
            'right_turn_start': [],
            'right_turn_end'  : [],
            }
        threshold = 2
        previous_event = 0

        timed_events = []
        start_times = 0 
        for i in xrange(threshold-1, len(Y)):
            x = Y[i]
            lower = max(0, i - (threshold))
            upper = min(len(Y), i + (threshold))
            if previous_event == 0 and \
                    x == 1 and \
                    Counter(Y[i:upper])[1] == threshold : # start left 
                print 'Start left'
                previous_event = 1
                events['left_turn_start'].append(i)
                start_times = df.iloc[i]['timestamp_x']

            if previous_event == 1 and\
                    x == 2 and\
                    Counter(Y[lower:i])[2] == threshold: # end left 
                print 'End left'
                events['left_turn_end'].append(i)
                previous_event = 0
                timed_events.append((
                    start_times,
                    df.iloc[i]['timestamp_x'],
                    'left_turn'
                    ))
            if previous_event == 0 and \
                    x == 3 and \
                    Counter(Y[i:upper])[3] == threshold : # start left 
                print 'Start right'
                previous_event = 3
                events['right_turn_start'].append(i)
                start_times = df.iloc[i]['timestamp_x']

            if previous_event == 3 and\
                    x == 4 and\
                    Counter(Y[lower:i])[4] == threshold: # end left 
                print 'End right'
                events['right_turn_end'].append(i)
                previous_event = 0
                timed_events.append((
                    start_times,
                    df.iloc[i]['timestamp_x'],
                    'right_turn'
                    ))

        return events, timed_events






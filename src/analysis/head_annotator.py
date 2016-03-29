from event_annotator import EventAnnotator
import os

import head_features
import pandas as pd
# from util import generate_windows
from sklearn.externals import joblib
from collections import Counter
import json


class HeadAnnotator(EventAnnotator):
    def __init__(self):
        self.events = []
        m_dir = os.path.dirname(__file__)
        base_dir = os.path.join(m_dir, '../models/head_turns/')
        self.model = joblib.load('%s/head_turns.pkl' % base_dir) 
        #base_dir = os.path.join(m_dir, '../models/head_turns_v2/')
        #self.model = joblib.load('%s/head_turns_v2.pkl' % base_dir) 
        print self.model.__class__
        config_fi = open('%s/config.json' % base_dir, 'r')
        self.config = json.loads(reduce(lambda x, y: x + y, config_fi))
        self.window_size = self.config['window_size']
        self.relevant_features = self.config['relevant_features']
        self.active_features = \
                pd.read_csv('%s/active_features.csv' % base_dir)['columns'].tolist()
        self.events=[]

    def annotate_events(self, df):
        window_size = self.window_size
        active_features = self.active_features
        og_cols = df.columns.tolist()
        df_feat, features = head_features.apply_feature_engineering(
                df,
                relevant_features=self.relevant_features,
                )
        df_feat.fillna(0, inplace=True)
        df_w, feats = head_features.generate_windows(
            df_feat, 
            window = window_size,
            relevant_features=features,
            )
        # Cut off the tail end of the data (lots of null values)
        df_w = df_w.loc[0:(len(df_w)-window_size)]
        Y = self.model.predict(df_w[active_features])
        df_w['class'] = Y
        for c in og_cols:
            if not c in active_features:
                df_w[c] = df[c]

        self.df = df_w
        print Y
        # These are the raw events
        return self.find_true_events(df, Y.tolist(), index_col='frameIndex')

    def find_true_events(self, df, Y, index_col = 'timestamp_x'):
        """
        This is a hacky way to combine classified signal points
        to generate signals
        """
        events = {
            'left_turn_start': [],
            'left_turn_end'  : [],
            'right_turn_start': [],
            'right_turn_end'  : [],
            }
        threshold = 2
        event_thresh = 15 
        timed_events = []
        min_event_length = 2 # minumn length event must be


        def end_event(event_key, start_index, end_index):
            if event_key in set([1,3]) and end_index - start_index >= min_event_length:
                if event_key == 1:
                    event_start = 'left_turn_start'
                    event_end   = 'left_turn_end'
                    event = 'left_turn'
                elif event_key == 3:
                    event_start = 'right_turn_start'
                    event_end   = 'right_turn_end'
                    event = 'right_turn'



                events[event_start].append(start_index)
                events[event_end].append(end_index)
                timed_events.append((
                    start_index,
                    end_index,
                    event,
                    ))



        previous_event = 0 # 0, 1 or 3
        potential_end_event = -1

        steps_since_start = 0

        start_index = 0
        for i in xrange(threshold-1, len(Y)):
            x = Y[i]
            lower = max(0, i - (threshold))
            upper = min(len(Y), i + (threshold))

            current_point = Y[i]
            # starting events are 1 or 3
            potentially_new_starting_event = \
                    Counter(Y[i:upper])[1] == threshold or \
                    Counter(Y[i:upper])[3] == threshold
            # ending events are 2 or 4
            potentially_new_ending_event = \
                    Counter(Y[lower:i])[2] == threshold or \
                    Counter(Y[lower:i])[4] == threshold


            if potentially_new_starting_event:
                if previous_event == 0:
                    # true new event
                    start_index = i
                    previous_event = current_point
                    steps_since_start = 0
                    potential_end_event = -1

                elif steps_since_start > event_thresh:
                    # the previous found events ending is strange and should be ended
                    if previous_event != 0:
                        # end the current event. there's
                        # probably a bug, but at least
                        # we can catch the event and not break new
                        # events
                        end_event(
                                previous_event, 
                                start_index, 
                                # either end now or a previously 
                                # found event (if found)
                                #max(potential_end_event, i-1)
                                potential_end_event if potential_end_event != -1 \
                                    else start_index + event_thresh
                                )
                        previous_event = current_point
                        start_index = i
                        steps_since_start = 0
                        potential_end_event = -1
            elif potentially_new_ending_event:
                if previous_event == (current_point - 1):
                    end_event(
                            previous_event,
                            start_index,
                            i,
                            )
                    previous_event = 0
                    steps_since_start = 0
                    potential_end_event = -1
                
            else:
                if previous_event != 0:
                    steps_since_start += 1
                    if potential_end_event == -1 and \
                            previous_event == (current_point-1):
                        # the event could end here
                        potential_end_event = i
        if previous_event != 0:
            end_event(
                    previous_event,
                    start_index,
                    min(
                        len(Y),
                        potential_end_event if potential_end_event != -1 \
                                else (start_index + event_thresh)
                    
                        )
                    )
        self.events = timed_events
        return events, timed_events






import os

from sklearn.externals import joblib
import pandas as pd

from signal_classifier import SignalClassifier 
from head_signal_sentiment_features import compute_signal_features

class SignalHeadClassifier(SignalClassifier):
    def __init__(self, df, signal_indices):
        SignalClassifier.__init__(self, df, signal_indices)
        m_dir = os.path.dirname(__file__)
        base_dir = os.path.join(m_dir, '../models/head_sentiment/')
        model_dir = os.path.join(base_dir, 'head_sentiment_v0')
        self.model = joblib.load('%s/head_sentiment_v0.pkl' % model_dir) 

        self.active_features = \
                pd.read_csv('%s/active_features.csv' % model_dir)['columns'].tolist()

    def classify_signals(self):
        events = []
        print self.signal_indices
        print 'DF length', len(self.df)
        for event in self.signal_indices:
            start_index = int(event[0])
            end_index = int(event[1])
            end_index = len(self.df)-1 if end_index > len(self.df) else end_index
            event_name = event[2]
            df_sub = self.df.loc[start_index:end_index]
            events.append(compute_signal_features(df_sub))
        # only print if there is actual events
        if len(events) > 0:
            df_events = pd.DataFrame(events)
            return self.model.predict(df_events[self.active_features])
        return []



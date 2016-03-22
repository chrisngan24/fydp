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

import numpy.fft as fft

import statistics

class LaneAnnotator(EventAnnotator):
    def __init__(self, data_direc):
        m_dir = os.path.dirname(__file__)
        self.base_dir = os.path.join(m_dir, '../models/lane_changes/')
        self.data_direc = os.path.join(m_dir, '../%s' %data_direc)

        self.left = []
        self.right = []
        self.left_turn = []
        self.right_turn = []
        self.neg = []

        dtw_models_direc = os.path.join(self.base_dir, 'lane_changes')

        for subdir, dirs, files in os.walk(dtw_models_direc):
            for d in dirs:
                if d.startswith("left_") and not d.startswith("left_turn"):
                    self.left.append(pd.read_csv("%s/model.csv" %os.path.join(dtw_models_direc, d)))
                elif d.startswith("right_") and not d.startswith("right_turn"):
                    self.right.append(pd.read_csv("%s/model.csv" %os.path.join(dtw_models_direc, d)))
                else:
                    self.neg.append(pd.read_csv("%s/model.csv" %os.path.join(dtw_models_direc, d)))

        self.model = joblib.load('%s/knn.pkl' % self.base_dir) 
        config_fi = open('%s/config.json' % self.base_dir, 'r')
        self.config = json.loads(reduce(lambda x, y: x + y, config_fi))
        self.window_size = self.config['window_size']
        self.ignore_columns = self.config['ignore_columns']
        self.active_features = self.config['active_features']
        self.moving_average_size = self.config['moving_average_size']

        self.events = []

    def is_valid_event(self, signal, start_index, end_index, e_type):
        if len(signal) < 60:
            return False

        if max(signal) - min(signal) < 20:
            return False

        left_costs = [fastdtw(signal, x['theta'].tolist())[0] for x in self.left]
        right_costs = [fastdtw(signal, x['theta'].tolist())[0] for x in self.right]
        neg_costs = [fastdtw(signal, x['theta'].tolist())[0] for x in self.neg]

        min_left_cost = min(left_costs)
        min_right_cost = min(right_costs)
        min_neg_cost = min(neg_costs)

        median_left_cost = statistics.median(left_costs)
        median_right_cost = statistics.median(right_costs)
        median_neg_cost = statistics.median(neg_costs)

        self.stats_file.write("start and end indices: ")
        self.stats_file.write(str(start_index) + " " + str(end_index))
        self.stats_file.write("\n")

        self.stats_file.write("signal length: ")
        self.stats_file.write(str(len(signal)))
        self.stats_file.write("\n")

        self.stats_file.write("signal max and min: ")
        self.stats_file.write(str(max(signal)) + " " + str(min(signal)))
        self.stats_file.write("\n")

        self.stats_file.write("location of signal max and min: ")
        self.stats_file.write(str(signal.index(max(signal))) + " " + str(signal.index(min(signal))))
        self.stats_file.write("\n")

        self.stats_file.write("input event type: ")
        self.stats_file.write(e_type)
        self.stats_file.write("\n")

        self.stats_file.write("min left: ")
        self.stats_file.write(str(min_left_cost))
        self.stats_file.write("\n")

        self.stats_file.write("min left index: ")
        left_index = left_costs.index(min_left_cost)
        self.stats_file.write(str(left_index))
        self.stats_file.write("\n")

        self.stats_file.write("min right: ")
        self.stats_file.write(str(min_right_cost))
        self.stats_file.write("\n")

        self.stats_file.write("min right index: ")
        right_index = right_costs.index(min_right_cost)
        self.stats_file.write(str(right_index))
        self.stats_file.write("\n")

        self.stats_file.write("min neg: ")
        self.stats_file.write(str(min_neg_cost))
        self.stats_file.write("\n")

        self.stats_file.write("min neg index: ")
        neg_index = neg_costs.index(min_neg_cost)
        self.stats_file.write(str(neg_index))
        self.stats_file.write("\n")

        self.stats_file.write("average of 3 min lefts: ")
        self.stats_file.write(str(statistics.mean(sorted(left_costs)[:3])))
        self.stats_file.write("\n")

        self.stats_file.write("average of 3 min rights: ")
        self.stats_file.write(str(statistics.mean(sorted(right_costs)[:3])))
        self.stats_file.write("\n")

        self.stats_file.write("average of 3 min negs: ")
        self.stats_file.write(str(statistics.mean(sorted(neg_costs)[:3])))
        self.stats_file.write("\n")

        self.stats_file.write("median left: ")
        self.stats_file.write(str(median_left_cost))
        self.stats_file.write("\n")

        self.stats_file.write("median right: ")
        self.stats_file.write(str(median_right_cost))
        self.stats_file.write("\n")

        self.stats_file.write("median neg: ")
        self.stats_file.write(str(median_neg_cost))
        self.stats_file.write("\n")

        self.stats_file.write("min left signal length: ")
        self.stats_file.write(str(len(self.left[left_index])))
        self.stats_file.write("\n")

        self.stats_file.write("min right signal length: ")
        self.stats_file.write(str(len(self.right[right_index])))
        self.stats_file.write("\n")

        self.stats_file.write("min neg signal length: ")
        self.stats_file.write(str(len(self.neg[neg_index])))
        self.stats_file.write("\n")

        spectrum = fft.fft(signal)
        max_frequency = max(abs(spectrum))
        median_frequency = statistics.median(abs(spectrum))

        self.stats_file.write("max frequency: ")
        self.stats_file.write(str(max_frequency))
        self.stats_file.write("\n")

        self.stats_file.write("median frequency: ")
        self.stats_file.write(str(median_frequency))
        self.stats_file.write("\n")

        self.stats_file.write("classified type: ")
        if min_left_cost < min_right_cost and min_left_cost < min_neg_cost and signal.index(max(signal)) < signal.index(min(signal)) and abs(max(signal) - signal[0]) + 30 > abs(signal[0] - min(signal)):
            self.stats_file.write("left")
        elif min_right_cost < min_left_cost and min_right_cost < min_neg_cost and signal.index(min(signal)) < signal.index(max(signal)) and abs(signal[0] - min(signal)) + 30 > abs(max(signal) - signal[0]):
            self.stats_file.write("right")
        else:
            self.stats_file.write("neg")
        self.stats_file.write("\n")
        self.stats_file.write("\n")

        if e_type == 'left':
            return min_left_cost < min_right_cost and min_left_cost < min_neg_cost and signal.index(max(signal)) < signal.index(min(signal)) and abs(max(signal) - signal[0] + 30) > abs(signal[0] - min(signal))
        else:
            return min_right_cost < min_left_cost and min_right_cost < min_neg_cost and signal.index(min(signal)) < signal.index(max(signal)) and abs(signal[0] - min(signal)) + 30 > abs(max(signal) - signal[0])

        # self.stats_file.write("classified type: ")
        # if median_left_cost < median_right_cost and median_left_cost < median_neg_cost:
        #     self.stats_file.write("left")
        # elif median_right_cost < median_left_cost and median_right_cost < median_neg_cost:
        #     self.stats_file.write("right")
        # else:
        #     self.stats_file.write("neg")
        # self.stats_file.write("\n")
        # self.stats_file.write("\n")

        # if e_type == 'left':
        #     return median_left_cost < median_right_cost and median_left_cost < median_neg_cost
        # else:
        #     return median_right_cost < median_left_cost and median_right_cost < median_neg_cost


    def annotate_events(self, df, index_col='frameIndex'):
        self.stats_file = open('%s/stats.txt' % self.data_direc, 'a')
        self.stats_file.seek(0)
        self.stats_file.truncate()

        df['gz'] = util.movingaverage(df['gz'], self.moving_average_size)
        windowed_df_test = util.generate_windows(df, window=self.window_size, ignore_columns=self.ignore_columns)
        windowed_df_test = windowed_df_test[self.active_features]

        predicted_labels_test = self.model.predict(windowed_df_test)

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
                        if self.is_valid_event(signal, k, i, 'left'):
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
                        if self.is_valid_event(signal, k, i, 'right'):
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

        self.stats_file.close()
        
        print events_indices
        
        return events, events_indices



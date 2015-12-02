from event_annotator import EventAnnotator
import os
import numpy as np
import dtw

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

        window_size_file = open('%s/average_lane_change_length.txt' % self.base_dir)
        average_sizes = window_size_file.read().split('\n')

        self.left_window_size = int(average_sizes[0])
        self.right_window_size = int(average_sizes[1])

    def annotate_events(self, df):
        events_hash, event_indices = dtw.find_start_end_indices(self.left_models, self.right_models, df)
        print events_hash
        print event_indices
        return events_hash, event_indices


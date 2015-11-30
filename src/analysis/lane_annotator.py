from event_annotator import EventAnnotator
import os
import numpy as np
import dtw

class LaneAnnotator(EventAnnotator):
    def __init__(self):
        m_dir = os.path.dirname(__file__)
        self.base_dir = os.path.join(m_dir, '../models/lane_changes/')

        window_size_file = open('%s/average_lane_change_length.txt' % self.base_dir)
        average_sizes = window_size_file.read().split('\n')

        left_model_file = open('%s/left_lane_change.txt' % self.base_dir)
        left_model_file_lines = left_model_file.read().split('\n')

        right_model_file = open('%s/right_lane_change.txt' % self.base_dir)
        right_model_file_lines = right_model_file.read().split('\n')

        self.left_window_size = int(average_sizes[0])
        self.right_window_size = int(average_sizes[1])
        self.left_model = np.array([float(i) for i in left_model_file_lines])
        self.right_model = np.array([float(i) for i in right_model_file_lines])

    def annotate_events(self, df):
        best_left = dtw.find_start_end_indices(self.left_model, df, self.left_window_size)
        best_right = dtw.find_start_end_indices(self.right_model, df, self.right_window_size)
        events_hash = { 'left_lane_change_start': [best_left[0]], 
        				'left_lane_change_end': [best_left[1]],
        				'right_lane_change_start': [best_right[0]], 
        				'right_lane_change_end': [best_right[1]]
        			  }
        print events_hash
        return events_hash

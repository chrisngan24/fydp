import pandas as pd
import numpy as np
import dtw

class Analysis:
	def __init__(self, model_dir, data_dir):
		self.model_dir = model_dir
		self.data_dir = data_dir

	def run(self, algorithm):

		events_hash = {}

		window_size_file = open('%s/average_lane_change_length.txt' % self.model_dir)
	
		average_sizes = window_size_file.read().split('\n')
		left_window_size = int(average_sizes[0])
		right_window_size = int(average_sizes[1])

		left_model_file = open('%s/left_lane_change.txt' % self.model_dir)
		left_model_file_lines = left_model_file.read().split('\n')
		left_model = np.array([float(i) for i in left_model_file_lines])

		right_model_file = open('%s/right_lane_change.txt' % self.model_dir)
		right_model_file_lines = right_model_file.read().split('\n')
		right_model = np.array([float(i) for i in right_model_file_lines])

		# data_file = open('%s/fused.csv' % self.data_dir)
		data_file = open('data/%s/fused.csv' % '1448660313')

		df = pd.read_csv(data_file, header=0, usecols=["theta"])

		# TODO: make dtw a function call
		best_left = dtw.find_start_end_indices(left_model, df, left_window_size)
		best_right = dtw.find_start_end_indices(right_model, df, right_window_size)
		
		events_hash = { 'left': best_left, 'right': best_right }

		return events_hash

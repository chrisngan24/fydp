import pandas as pd
import numpy as np
import dtw

class Analysis:
	def __init__(self, model_dir, data_dir):
		self.model_dir = model_dir
		self.data_dir = data_dir

	def run(self, algorithm):

		events_hash = {}

		model_file = open('%s/left_lane_change.txt' % self.model_dir)
		model_file_lines = model_file.read().split('\n')
		model = np.array([float(i) for i in model_file_lines])

		data_file = open('%s/fused.csv' % self.data_dir)

		df = pd.read_csv(data_file, header=0, usecols=["theta"])

		# TODO: make this a function call
		if algorithm == 'dtw':
			events_hash = dtw.find_start_end_indices(model, df)

		print events_hash

		return events_hash

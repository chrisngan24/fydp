import os.path as path
import pandas as pd
import numpy as np
from scipy import signal

data_direc = 'data'
output_direc = 'output'

data_file_name = 'fused.csv'
label_file_name = 'labels.txt'

left_lane_change_model_file_name = 'left_lane_change'
right_lane_change_model_file_name = 'right_lane_change'

average_lane_change_length_file_name = 'average_lane_change_length.txt'

def read_data():
	data_file = open(path.join(data_direc, data_file_name))
	labels_file = open(path.join(data_direc, label_file_name))

	df = pd.read_csv(data_file, header=0, usecols=["theta"])

	labels_file_lines = labels_file.read().split('\n')
	labels = np.array([i.split(',') for i in labels_file_lines])

	return [df, labels]

def train(df, labels):
	average_left_length = 0
	average_right_length = 0
	num_left = 0

	left_models = []
	right_models = []

	max_length = max(int(row[1]) - int(row[0]) for row in labels)

	for start, end, label in labels:
		start = int(start)
		end = int(end)
		lane_change_data = pd.np.array(df[start:end])
		# lane_change_data = signal.resample(lane_change_data, max_length)
		
		if label == 'left':
			average_left_length += end - start
			num_left += 1
			left_models.append(lane_change_data)
		else:
			average_right_length += end - start
			right_models.append(lane_change_data)

	average_left_length /= num_left
	average_right_length /= len(labels) - num_left
	
	average_length_string = '\n'.join([str(average_left_length), str(average_right_length)])

	for i in xrange(len(left_models)):
		m = left_models[i].tolist()
		m_flat = [str(val) for sublist in m for val in sublist]
		save_model('\n'.join(m_flat), path.join(output_direc, 'left_%s.txt' % str(i)))

	for j in xrange(len(right_models)):
		m = right_models[j].tolist()
		m_flat = [str(val) for sublist in m for val in sublist]
		save_model('\n'.join(m_flat), path.join(output_direc, 'right_%s.txt' % str(j)))

	save_model(average_length_string, path.join(output_direc, average_lane_change_length_file_name))

def save_model(text, file_name):
    target = open(file_name, 'w')
    if text[-1:] == '\n':
        text = text[:-1]
    target.write(text)

def generate_windows(df, window=10, relative_columns = None):
    points = []
    cols = relative_columns
    if relative_columns == None:
        cols = df.columns.values
    for i, r in df.iterrows():
        w_start = i
        w_end   = min(i + 100, len(df)-1)
        row = r.to_dict()
        df_w = df.loc[w_start:w_end].reset_index(drop=True)
        for j in xrange(0,window):
            if j < len(df_w):
                window_row = df_w.loc[j].to_dict()
            else:
                window_row = None
            for c in cols:
                name = '%s_%s' % (c, j)
                row[name] = window_row[c] if window_row != None else None
        points.append(row)
        
    return pd.DataFrame(points)

df, labels = read_data()
train(df, labels)


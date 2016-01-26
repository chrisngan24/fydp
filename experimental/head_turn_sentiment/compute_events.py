"""
Reuse code from prod
Collect data from `face_modeling`
"""
import os
import sys
sys.path.append('../../src')

import pandas as pd


from analysis.head_annotator import HeadAnnotator

def compute_events(data_dir):
    for csv in os.listdir(data_dir):
        if not csv.find('.csv') == -1:
            fi_path = '%s/%s' % (data_dir, csv)
            df = pd.read_csv(fi_path)
            event_hash, event_list = HeadAnnotator().annotate_events(df)
            import pdb; pdb.set_trace()

if __name__ == '__main__':
    base_data_dir = '../face_modeling/data'
    data_files = sys.argv[1].split(',')
    for d in data_files:
        compute_events('%s/%s' % (base_data_dir, d))

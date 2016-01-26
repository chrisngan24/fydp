"""
Reuse code from prod
Collect data from `face_modeling`
"""
import os
import sys

import pandas as pd


sys.path.append('../../src')
from analysis.head_annotator import HeadAnnotator

sys.path.append('../face_modeling')
import head_features

def compute_events(data_dir, start_key, end_key, output_dir=None):
    if output_dir != None:
        make_dir(output_dir)

    df_merged = pd.DataFrame()
    for csv in os.listdir(data_dir):
        if not csv.find('.csv') == -1:
            fi_path = '%s/%s' % (data_dir, csv)
            df = pd.read_csv(fi_path)
            ha = HeadAnnotator() 
            event_hash, event_list = ha.annotate_events(df)
            df_p = ha.df
            og_cols = df.columns.tolist()
            fe_cols = df_p.columns.tolist()
            for c in og_cols:
                if not c in fe_cols:
                    df_p[c] = df[c]
            if output_dir != None:
                sub_dir = '%s/events' % output_dir
                make_dir(sub_dir)
                # assume start key and end key event s have the same
                for i in xrange(len(event_hash[start_key])):
                    start = event_hash[start_key][i]
                    end = event_hash[end_key][i]
                    df_sub = df_p.loc[start:end]
                    df_sub['original_index'] = df_sub.index
                    print csv
                    df_sub.to_csv('%s/%s-%s.csv' % (sub_dir, csv.split('.')[0], i), index=False)
            if output_dir != None:
                df_p.to_csv('%s/%s' % (output_dir, csv), index=False)
            df_merged = pd.concat([df_merged, df_p])
            import pdb; pdb.set_trace()
    return df_merged

def make_dir(m_dir):
    if not os.path.exists(m_dir) or \
            not os.path.isdir(m_dir):
        os.makedirs(m_dir)


if __name__ == '__main__':
    base_data_dir = '../face_modeling/data'
    data_files = sys.argv[1].split(',')
    output_dir = 'data'
    merged_dir = '%s/merged' % output_dir
    make_dir(output_dir)
    make_dir(merged_dir)

    for d in data_files:
        read_dir ='%s/%s' % (base_data_dir, d)
        write_dir = '%s/%s' % (output_dir, d)
        df = pd.DataFrame()
        if d.find('left') != -1:
            df = compute_events(
                    read_dir,
                    'left_turn_start',
                    'left_turn_end',
                    output_dir=write_dir,
                    )
        elif d.find('right') != -1:
            df = compute_events(
                    read_dir,
                    'right_turn_start',
                    'right_turn_end',
                    output_dir=write_dir,
                    )

        df.to_csv('%s/%s.csv' % (merged_dir, d), index=False)
        import pdb; pdb.set_trace()
        

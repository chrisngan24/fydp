import cv2
import logging

from analysis.head_annotator import HeadAnnotator
from analysis.lane_annotator import LaneAnnotator
import annotation
import pandas as pd
import numpy as np
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def run_single_test(case_name, results_df):

    # Read everything you need
    df = pd.read_csv('test_suite/test_cases/' + case_name + '/fused.csv')
    max_index = max(df['frameIndex'])
    baseline = eval(open('test_suite/test_cases/' + case_name + '/annotation_josh.txt', 'r').read())

    # Declare storage for annotated frames
    baseline_frames = dict(
        left_turn=np.zeros(max_index, dtype=np.int8), 
        right_turn=np.zeros(max_index, dtype=np.int8), 
        left_lane_change=np.zeros(max_index, dtype=np.int8), 
        right_lane_change=np.zeros(max_index, dtype=np.int8))
    
    # For each event, mark in the baseline
    i = 0
    while (i < len(baseline)):
        start = baseline[i]['start']
        end = baseline[i]['end']
        event_type = baseline[i]['type']
        baseline_frames[event_type][start:end] += 1
        i += 1

    # Use the annotation code to generate an event list
    head_events_hash, head_events_list = HeadAnnotator().annotate_events(df)
    lane_events_hash, lane_events_list = LaneAnnotator().annotate_events(df)

    i = 0
    while (i < len(head_events_list)):
        start = head_events_list[i][0]
        end = head_events_list[i][1]
        event_type = head_events_list[i][2]
        baseline_frames[event_type][start:end] += 1
        i += 1        

    i = 0
    while (i < len(lane_events_list)):
        start = lane_events_list[i][0]
        end = lane_events_list[i][1]
        event_type = lane_events_list[i][2]
        baseline_frames[event_type][start:end] += 1
        i += 1     

    wrong_count_left = np.shape(np.where(baseline_frames['left_turn'] == 1))[1]
    wrong_count_right = np.shape(np.where(baseline_frames['right_turn'] == 1))[1]
    wrong_count_left_lane = np.shape(np.where(baseline_frames['left_lane_change'] == 1))[1]
    wrong_count_right_lane = np.shape(np.where(baseline_frames['right_lane_change'] == 1))[1]

    test_results = dict(
        case_name=case_name,
        left_turn=round(1 - float(wrong_count_left) / max_index, 3),
        right_turn=round(1 - float(wrong_count_right) / max_index, 3),
        left_lane_change=round(1 - float(wrong_count_left_lane) / max_index, 3),
        right_lane_change=round(1 - float(wrong_count_right_lane) / max_index, 3)
        )

    results_df.loc[case_name] = test_results

if __name__ == '__main__':
    
    print "Running Tests...."

    results_df = pd.DataFrame(columns=['case_name', 'left_turn', 'right_turn', 'left_lane_change', 'right_lane_change'])

    output_file = open("test_results.html", 'w')
    test_case_list = sorted(next(os.walk('test_suite/test_cases/'))[1])
    for test in test_case_list:
        run_single_test(test, results_df)

    output_file.write(results_df.to_html())


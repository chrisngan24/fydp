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

def run_single_test(case_name, results_list, annotation_file = 'annotation_josh.txt', testing_dir='test_suite/test_cases/'):
    print case_name
    print annotation_file 
    # Read everything you need
    df = pd.read_csv(testing_dir + case_name + '/fused.csv')
    max_index = max(df['frameIndex'])
    baseline = eval(open(testing_dir + case_name + '/' + annotation_file, 'r').read())

    # Declare storage for annotated frames
    baseline_frames = dict(
        left_turn=np.zeros(max_index, dtype=np.int8), 
        right_turn=np.zeros(max_index, dtype=np.int8), 
        left_lane_change=np.zeros(max_index, dtype=np.int8), 
        right_lane_change=np.zeros(max_index, dtype=np.int8))
    
    # For each event, mark in the baseline
    for i in xrange(len(baseline)):
        start = baseline[i]['start']
        end = baseline[i]['end']
        event_type = baseline[i]['type']
        baseline_frames[event_type][start:end] += 1

    # Use the annotation code to generate an event list
    head_events_hash, head_events_list = HeadAnnotator().annotate_events(df)
    lane_events_hash, lane_events_list = LaneAnnotator().annotate_events(df)

    for i in xrange(len(head_events_list)):
        start = head_events_list[i][0]
        end = head_events_list[i][1]
        event_type = head_events_list[i][2]
        baseline_frames[event_type][start:end] += 1

    for i in xrange(len(lane_events_list)):
        start = lane_events_list[i][0]
        end = lane_events_list[i][1]
        event_type = lane_events_list[i][2]
        baseline_frames[event_type][start:end] += 1

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
    results_list.append(test_results)


def main():
    print "Running Tests...."
    testing_dir = 'test_suite/test_cases/'


    #results_df = pd.DataFrame(columns=['case_name', 'left_turn', 'right_turn', 'left_lane_change', 'right_lane_change'])
    results_list = []

    output_file = open("test_results.html", 'w')
    test_case_list = sorted(next(os.walk(testing_dir))[1])
    print test_case_list
    for test in test_case_list:
        for fi in os.listdir(testing_dir + test):
            # hacky but yolo
            if fi.find('drivelog_temp_annotated_') == 0 or fi.find('annotation_') == 0:
                run_single_test(test, results_list, annotation_file=fi)
    results_df = pd.DataFrame(results_list)
    output_file.write(results_df.to_html())



if __name__ == '__main__':
    main() 
    

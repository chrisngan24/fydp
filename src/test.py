import cv2
import logging
import copy

from analysis.head_annotator import HeadAnnotator
from analysis.lane_annotator import LaneAnnotator
from optparse import OptionParser

import runner 
import annotation

import pandas as pd
import numpy as np
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def run_single_test(
        case_name,
        build_name, 
        results_list, 
        event_types = [
            'left_turn', 
            'right_turn', 
            'left_lane_change', 
            'right_lane_change',
            ],
        annotation_file = 'annotation_josh.txt', 
        testing_dir='test_suite/test_cases/'):
    print case_name
    print annotation_file 
    # Read everything you need
    path_to_test_video = testing_dir + case_name 
    # df = pd.read_csv(path_to_test_video + '/fused.csv')
    print build_name
    if (build_name == None):
        files = [ path_to_test_video + '/WHEEL.csv',
                  path_to_test_video + '/CAMERA.csv',
                  ]
    else:
        files = [ path_to_test_video + '/WHEEL.csv',
                  path_to_test_video + '/CAMERA-' + build_name + '.csv',
                  ]    

    analysis_results = runner.run_fusion(
            files, 
            has_camera=True,
            has_wheel=True,
            data_direc=path_to_test_video,
            is_move_video=False,
            )
    df = analysis_results['df']
    # Use frame index from the test results
    max_index = max(df['frameIndex'])
    baseline = eval(open(testing_dir + case_name + '/' + annotation_file, 'r').read())

    # Declare storage for annotated frames
    zero_frames = { 
            key : np.zeros(max_index, dtype=np.int8) \
                    for key in event_types
                    }

    ####
    # Frames will have either a 0 or 1 at each index
    # 0 indicates that no event is either annotated or predicted
    #   at the index
    # 1 indicates that an event is predicted or annotated
    #   at the index
    event_frames = copy.deepcopy(zero_frames)
    annotation_frames =  copy.deepcopy(zero_frames)
    # For each event, mark in the baseline
    for i in xrange(len(baseline)):
        start = baseline[i]['start']
        end = baseline[i]['end']
        event_type = baseline[i]['type']
        annotation_frames[event_type][start:end] += 1

    # Use the annotation code to generate an event list
    '''
    head_events_hash, head_events_list = HeadAnnotator().annotate_events(df)
    lane_events_hash, lane_events_list = LaneAnnotator().annotate_events(df)
    '''
    head_events_list = analysis_results['head_events_list']
    lane_events_list = analysis_results['lane_events_list']

    predicted_events_list = head_events_list + lane_events_list
    for i in xrange(len(predicted_events_list)):
        start = int(df.iloc[predicted_events_list[i][0]]['frameIndex'])
        end = int(df.iloc[predicted_events_list[i][1]]['frameIndex'])
        event_type = predicted_events_list[i][2]
        event_frames[event_type][start:end] += 1


    event_summaries=dict()
    for event in event_types:
        # indices where prediction of the event was made
        predicted_frame_index = np.where(event_frames[event] == 1)[0]
        # indices where prediction of the event was not made
        not_predicted_frame_index = np.where(event_frames[event] == 0)[0]
        
        annotations = annotation_frames[event]
        predictions = event_frames[event]

        event_summaries['wrong_count_%s' % event] = \
            sum((annotations + predictions) == 1) 
        event_summaries['tp_count_%s' % event] = \
            sum(annotations[predicted_frame_index] == 1)
        event_summaries['fp_count_%s' % event] = \
            sum(annotations[predicted_frame_index] == 0)
        event_summaries['tn_count_%s' % event] = \
            sum(annotations[not_predicted_frame_index] == 0)
        event_summaries['fn_count_%s' % event] = \
            sum(annotations[not_predicted_frame_index] == 1)

    test_results = dict(
        case_name=case_name,
        )

    for event in event_types:
        wrong_count_key = 'wrong_count_%s' % event
        tp_count_key = 'tp_count_%s' % event
        fp_count_key = 'fp_count_%s' % event
        fn_count_key = 'fn_count_%s' % event
        wrong_count = event_summaries[wrong_count_key]
        tp_count = event_summaries[tp_count_key]
        fp_count = event_summaries[fp_count_key]
        fn_count = event_summaries[fn_count_key]
        test_results[event] = round(1 - float(wrong_count) / max_index, 3)
        test_results['%s_precision' % event] = \
                round(tp_count / float(max(tp_count + fp_count,1)), 3)
        test_results['%s_recall' % event] = \
                round(tp_count / float(max(tp_count + fn_count,1)), 3)
    results_list.append(test_results)


def main(build_name = None):
    print "Running tests"
    testing_dir = 'test_suite/test_cases/'

    #results_df = pd.DataFrame(columns=['case_name', 'left_turn', 'right_turn', 'left_lane_change', 'right_lane_change'])
    results_list = []

    output_file = open("test_results/test_results.html", 'w')
    test_case_list = sorted(next(os.walk(testing_dir))[1])
    print test_case_list
    for test in test_case_list:
        for fi in os.listdir(testing_dir + test):
            # hacky but yolo
            if fi.find('drivelog_temp_annotated_') == 0 or fi.find('annotation_') == 0:
                run_single_test(test, build_name, results_list, annotation_file=fi)

    results_df = pd.DataFrame(results_list)
    
    # Add the build name to the output
    if (build_name != None):
        output_file.write('BuildName: ' + build_name)
    
    output_file.write(results_df.to_html())
    output_file.write('<br/>\n<br/>\n<br/>')
    output_file.write('Summary:')
    output_file.write(results_df.describe().to_html())

# @clement: This is the function I used to copy all the things over. Committing it once for record. NEVER CALL THIS EVER AGAIN
def add_sentiment():

    # Adding a death line here
    exit()

    testing_dir = 'test_suite/test_cases/'
    test_case_list = sorted(next(os.walk(testing_dir))[1])
    print test_case_list
    for test in test_case_list:
        for fi in os.listdir(testing_dir + test):
            # hacky but yolo
            if fi.find('drivelog_temp_annotated_') == 0 or fi.find('annotation_') == 0:
                filename = testing_dir + test + '/' + fi
                baseline = eval(open(filename, 'r').read())
                file_event_list = []
                
                for i in xrange(len(baseline)):
                    start = baseline[i]['start']
                    end = baseline[i]['end']
                    event_type = baseline[i]['type']
                    event = dict(start = start, end = end, type = event_type, is_good = True)
                    file_event_list.append(event)

                f = open(testing_dir + test + '/modified_' + fi, 'w')
                f.write(str(file_event_list))

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-b', '--BuildName', default=None)
    (options, args) = parser.parse_args()

    main(options.BuildName) 
    

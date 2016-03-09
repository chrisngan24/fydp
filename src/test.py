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

def map_sentiment(col_name, is_good):
    if is_good:
        return col_name + '_good'
    else:
        return col_name + '_bad'

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
    # modify event types to include sentiment:
    event_types = reduce(
            lambda x,y: x + y,
            map(
                lambda x: [x[0], x[0] + '_bad', x[0] + '_good'], 
                map(lambda x: [x], event_types)
                )
            )
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
            is_interact=False,
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

    # The results from algorithm annotation
    event_frames = copy.deepcopy(zero_frames)
    # The results from human annotations 
    annotation_frames =  copy.deepcopy(zero_frames)
    # For each event, mark in the baseline
    for i in xrange(len(baseline)):
        start = baseline[i]['start']
        end = baseline[i]['end']
        event_type = baseline[i]['type']
        is_good = baseline[i]['is_good']
        annotation_frames[event_type][start:end] += 1
        ## Lets add a new column for just general sentiment
        annotation_frames[map_sentiment(event_type, is_good)][start:end] += 1



    # Use the annotation code to generate an event list
    head_events_list = analysis_results['head_events_list']
    lane_events_list = analysis_results['lane_events_list']
    head_event_sentiment_list = analysis_results['head_events_sentiment']
    lane_event_sentiment_list = analysis_results['lane_events_sentiment']

    predicted_events_list = head_events_list + lane_events_list
    sentiment_events_list = head_event_sentiment_list + lane_event_sentiment_list 
    for i in xrange(len(predicted_events_list)):
        start = int(df.iloc[predicted_events_list[i][0]]['frameIndex'])
        end = int(df.iloc[predicted_events_list[i][1]]['frameIndex'])
        event_type = predicted_events_list[i][2]
        is_good = sentiment_events_list[i][0]
        event_frames[event_type][start:end] += 1
        event_frames[map_sentiment(event_type,is_good)][start:end] += 1


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
        annotation_file=annotation_file,
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
    output_dir = 'test_results/'
    #results_df = pd.DataFrame(columns=['case_name', 'left_turn', 'right_turn', 'left_lane_change', 'right_lane_change'])
    results_list = []

    output_file = open(output_dir + "test_results.html", 'w')
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
    results_df.to_csv(output_dir + 'test_results.csv', index=False)

    ## Need panda 0.17.1
    html = results_df.style.background_gradient(
            cmap='Spectral',
            )
    
    output_file.write(html.render())
    output_file.write('<br/>\n<br/>\n<br/>')
    output_file.write('Summary:')
    df_summary = results_df.describe()
    df_summary = df_summary.append([
        pd.Series(results_df.median(), name='median'),
        pd.Series(
            results_df._get_numeric_data().apply(
                lambda x: sum(x >= 0.8) / float(len(x)), 
                axis=0,),
            name='proportion_geq_0.8',
            )
        ])
    
    output_file.write(df_summary.to_html())


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-b', '--BuildName', default=None)
    (options, args) = parser.parse_args()

    main(options.BuildName) 
    

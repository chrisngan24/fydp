import cv2
import logging
import copy

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

def run_kappa(
        case_name,
        rater1, 
        rater2,
        event_types = [
            'left_turn', 
            'right_turn', 
            'left_lane_change', 
            'right_lane_change',
            ], 
        testing_dir='test_suite/test_cases/'):
    
    baseline_r1 = eval(open(testing_dir + case_name + '/drivelog_temp_annotated_' + rater1 + '.txt', 'r').read())
    baseline_r2 = eval(open(testing_dir + case_name + '/drivelog_temp_annotated_' + rater2 + '.txt', 'r').read())

    max_index = -1

    for i in xrange(len(baseline_r1)):
        end = baseline_r1[i]['end']
        if (end > max_index):
            max_index = end

    for i in xrange(len(baseline_r2)):
        end = baseline_r1[i]['end']
        if (end > max_index):
            max_index = end

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
    annotation_frames_r1 =  copy.deepcopy(zero_frames)
    annotation_frames_r2 =  copy.deepcopy(zero_frames)
    agreement_frames =  copy.deepcopy(zero_frames)
    
    # For each event, mark in the baseline
    for i in xrange(len(baseline_r1)):
        start = baseline_r1[i]['start']
        end = baseline_r1[i]['end']
        event_type = baseline_r1[i]['type']
        is_good = baseline_r1[i]['is_good']
        annotation_frames_r1[event_type][start:end] += 1

    # For each event, mark in the baseline
    for i in xrange(len(baseline_r2)):
        start = baseline_r2[i]['start']
        end = baseline_r2[i]['end']
        event_type = baseline_r2[i]['type']
        is_good = baseline_r2[i]['is_good']
        annotation_frames_r2[event_type][start:end] += 1

    event_summaries=dict()
    kappas = []
    for event in event_types:
        
        agreement_frames[event] = annotation_frames_r1[event] + annotation_frames_r2[event]
        rater1_yes = sum((annotation_frames_r1[event]) == 1)
        rater2_yes = sum((annotation_frames_r2[event]) == 1)
        agreements = sum(agreement_frames[event] == 2) + sum(agreement_frames[event] == 0)
        disagreements = sum((agreement_frames[event]) == 1)
        total = (agreements + disagreements)

        p0 = agreements / float(total)
        pd = disagreements / float(total)
        r1_prob_yes = rater1_yes / float(total)
        r2_prob_yes = rater2_yes / float(total)

        # Find the chance that the raters agree, by chance
        pe = (r1_prob_yes * r2_prob_yes) + ((1 - r1_prob_yes) * (1 - r2_prob_yes))
        kappa = (p0 - pe) / (1 - pe)
        event_summaries[event] = kappa
        kappas.append(kappa)

        '''
        print "r1 yes: " + str(rater1_yes)
        print "r2 yes: " + str(rater2_yes)
        print "both yes: " + str(agreements)
        print "disagreements: " + str(disagreements)
        print "total: " + str(total)
        print "p0: " + str(p0)
        print "pe: " + str(pe)
        print kappa
        '''

    #print case_name + " " + rater1 + " " + rater2
    #print event_summaries
    return (kappas, event_summaries)

def main():
    
    print "Calculate IRR"
    testing_dir = 'test_suite/test_cases/'
    output_dir = 'test_results/'
    
    rater_pairings_detailed = dict()
    kappa_list = dict()

    test_case_list = sorted(next(os.walk(testing_dir))[1])
    print test_case_list
    print "Total count: " + str(len(test_case_list))
 
    for test in test_case_list:
        
        # Need to find the two raters
        rater1 = None
        rater2 = None

        for fi in sorted(os.listdir(testing_dir + test)):
            if fi.find('drivelog_temp_annotated_') == 0:        
                # Remove the first 24 characters ('drivelog_temp_annotated_') and last 4 ('.txt')
                rater = fi[24:-4]
                if rater1 == None:
                    rater1 = rater
                else:
                    rater2 = rater

        if (rater1 != None and rater2 != None):
            
            (kappas, kappa_summary) = run_kappa(test, rater1, rater2)
            raters = rater1 + '_' + rater2

            if raters not in rater_pairings_detailed:
                rater_pairings_detailed[raters] = []
                kappa_list[raters] = []
            
            rater_pairings_detailed[raters].append(kappa_summary)
            kappa_list[raters] = kappa_list[raters] + kappas

        else:
            print "WARNING: " + test + " does not have at least 2 raters!" 

    for raters in kappa_list:
        average_kappa = sum(kappa_list[raters]) / len(kappa_list[raters])
        print raters + " kappa is: " + str(average_kappa)

if __name__ == '__main__':
    main() 
    

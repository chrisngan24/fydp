from signal_classifier import SignalClassifier 

class SignalLaneClassifier(SignalClassifier):
    def __init__(self, df, lane_indices, head_indices, head_events_hash, head_events_sentiment):
        SignalClassifier.__init__(self, df, lane_indices)
        self.head_indices = head_indices
        self.head_events_hash = head_events_hash
        self.head_events_sentiment = head_events_sentiment
    
    def classify_signals(self):
        """
        Classify the lane change with heuristic
        """
        time_thresh = 1 # second
        lane_events_sentiment = []
        lane_events_list = self.signal_indices
        head_events_list = self.head_indices
        head_events_hash = self.head_events_hash
        head_events_sentiment = self.head_events_sentiment
        df = self.df
        for lane in lane_events_list:
            start_index, end_index, event = lane
            # suitable time frame that turns end index can be in
            max_time = df.loc[int(start_index)]['time'] + time_thresh
            min_time = df.loc[int(start_index)]['time'] - time_thresh
            # all the possible indices of the dataframe that could have the right time stamp
            possible_indice = set(df[(df['time'] > min_time) & (df['time'] < max_time)].index)
            
            # hacky
            # match the <direction>_lane_change to <direction>_turn_end
            end_head_turn_event = event.replace('lane_change', 'turn_end')
            
            end_indice = set(head_events_hash[end_head_turn_event])
            # find the end head turns that match the time threshold
            intersecting_indice = possible_indice & end_indice
            if intersecting_indice:
                # incase there is more than one relevant event
                relevant_end_index = max(intersecting_indice) 
                # hacky
                # match the index values that have 
                # the same ending index and the right event
                head_event_indice= [i for i, (s, e, event) \
                                    in enumerate(head_events_list) \
                                    if int(e) == relevant_end_index \
                                    and event == end_head_turn_event.replace('_end', '')]


                if head_events_sentiment[head_event_indice[0]]:
                    print 'good lane change'
                    lane_events_sentiment.append((True, ''))
                else:
                    print 'bad head turn'
                    lane_events_sentiment.append((False, 'Bad head turn'))
                
            else:
                print 'missing head turn'
                lane_events_sentiment.append((False, 'Missing head turn'))  
        return lane_events_sentiment

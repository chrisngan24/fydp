import cv2
import numpy as np
import json

def add_side_borders(frame):
    black = np.zeros((240,120,3), dtype=np.uint8)
    left_image = np.concatenate((black,frame), axis=1)
    layered = np.concatenate((left_image,black), axis=1)
    return layered

def add_labels(layered):

    default_font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img = layered, 
            text = "Right Head", 
            org = (10,20), 
            fontFace = default_font, 
            fontScale = 0.5, 
            color = (255,255,255), 
            thickness = 1)

    cv2.putText(img = layered, 
                text = "Turn", 
                org = (25,40), 
                fontFace = default_font, 
                fontScale = 0.5, 
                color = (255,255,255), 
                thickness = 1)

    cv2.putText(img = layered, 
                text = "Left Head", 
                org = (450,20), 
                fontFace = default_font, 
                fontScale = 0.5, 
                color = (255,255,255), 
                thickness = 1)

    cv2.putText(img = layered, 
                text = "Turn", 
                org = (465,40), 
                fontFace = default_font, 
                fontScale = 0.5, 
                color = (255,255,255), 
                thickness = 1)

    cv2.putText(img = layered, 
                text = "Right Lane", 
                org = (10,140), 
                fontFace = default_font, 
                fontScale = 0.5, 
                color = (255,255,255), 
                thickness = 1)

    cv2.putText(img = layered, 
                text = "Change", 
                org = (25,160), 
                fontFace = default_font, 
                fontScale = 0.5, 
                color = (255,255,255), 
                thickness = 1)

    cv2.putText(img = layered, 
                text = "Left Lane", 
                org = (450,140), 
                fontFace = default_font, 
                fontScale = 0.5, 
                color = (255,255,255), 
                thickness = 1)

    cv2.putText(img = layered, 
                text = "Change", 
                org = (460,160), 
                fontFace = default_font, 
                fontScale = 0.5, 
                color = (255,255,255), 
                thickness = 1)

    return layered

def sent_to_text(sentiment):
    if (sentiment):
        return "Good!"
    else:
        return "Bad!"

def sent_to_colour(sentiment):
    if (sentiment):
        return (1,255,1)
    else:
        return (1,1,255)

def add_event_note(frame, event_type, sentiment):

    default_font = cv2.FONT_HERSHEY_SIMPLEX

    if (event_type == "right_turn"):

        # RIGHT HEAD
        cv2.rectangle(frame,(120,0),(440,240),sent_to_colour(sentiment),5)
        cv2.putText(img = frame, 
                    text = sent_to_text(sentiment), 
                    org = (15,80), 
                    fontFace = default_font, 
                    fontScale = 0.75, 
                    color = sent_to_colour(sentiment),
                    thickness = 2)

    if (event_type == "right_lane_change"):

        # RIGHT LANE
        cv2.rectangle(frame,(120,0),(440,240),sent_to_colour(sentiment),5)
        cv2.putText(img = frame, 
                    text = sent_to_text(sentiment), 
                    org = (15,200), 
                    fontFace = default_font, 
                    fontScale = 0.75, 
                    color = sent_to_colour(sentiment),
                    thickness = 2)

    if (event_type == "left_turn"):

        # LEFT HEAD
        cv2.rectangle(frame,(120,0),(440,240),sent_to_colour(sentiment),5)
        cv2.putText(img = frame, 
                    text = sent_to_text(sentiment), 
                    org = (450,80), 
                    fontFace = default_font, 
                    fontScale = 0.75, 
                    color = sent_to_colour(sentiment),
                    thickness = 2)

    if (event_type == "left_lane_change"):

        # LEFT LANE
        cv2.rectangle(frame,(120,0),(440,240),sent_to_colour(sentiment),5)
        cv2.putText(img = frame, 
                    text = sent_to_text(sentiment), 
                    org = (450,200), 
                    fontFace = default_font, 
                    fontScale = 0.75, 
                    color = sent_to_colour(sentiment),
                    thickness = 2)

    return frame

def annotate_video(input_name, output_name, head_events_list, lane_events_list, video_metadata_file):
    
    FRAME_RESIZE = (560,240)
    FOURCC = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    FRAME_RATE = 20

    cap = cv2.VideoCapture(input_name)
    out = cv2.VideoWriter(filename = output_name, fourcc = FOURCC, fps = FRAME_RATE, frameSize = FRAME_RESIZE)
    idx = 0

    # For creating a single typed video
    if (lane_events_list == None or head_events_list == None):

        # Figure out whether head or lane is the video
        if (head_events_list is not None):
            events_list = head_events_list
        elif (lane_events_list is not None):
            events_list = lane_events_list

        num_events = len(events_list)
        if (num_events == 0):
            return input_name

        # SO UGLY
        event_start_idx = events_list[0][0]
        event_end_idx = events_list[0][1]
        event_type = events_list[0][2]
        event_sentiment = events_list[0][3]

        event_idx = 0

        print events_list
        print event_sentiment

        while (cap.isOpened()):

            (ret, frame) = cap.read()
            
            if ret==False:
                break
                
            frame = add_side_borders(frame)
            frame = add_labels(frame)
            idx += 1

            if (idx > event_start_idx and idx < event_end_idx):
                frame = add_event_note(frame, event_type, event_sentiment)

            if (idx > event_end_idx):    
                event_idx += 1
            
                if (event_idx < num_events):
                                                
                    # Find new start and end
                    event_start_idx = events_list[event_idx][0]
                    event_end_idx = events_list[event_idx][1]
                    event_type = events_list[event_idx][2]
                    event_sentiment = events_list[event_idx][3]

            out.write(frame)

    # Making a fused video
    else:
        
        num_events = len(head_events_list) + len(lane_events_list)
        if (num_events == 0):
            return input_name

        while (cap.isOpened()):

            (ret, frame) = cap.read()
            
            if ret==False:
                break
                
            frame = add_side_borders(frame)
            frame = add_labels(frame)
            idx += 1

            for event in head_events_list:
                start_idx = event[0]
                end_idx = event[1]
                event_type = event[2]
                event_sentiment = event[3]

                if (idx > start_idx and idx < end_idx):
                    frame = add_event_note(frame, event_type, event_sentiment)

            for event in lane_events_list:
                start_idx = event[0]
                end_idx = event[1]
                event_type = event[2]
                event_sentiment = event[3]

                if (idx > start_idx and idx < end_idx):
                    frame = add_event_note(frame, event_type, event_sentiment)
                
            out.write(frame)
      
    video_metadata = dict()
    video_metadata['frames'] = idx
    video_metadata['fps'] = FRAME_RATE
    video_metadata['head_events'] = head_events_list
    video_metadata['lane_events'] = lane_events_list

    video_metadata_out = open(video_metadata_file, 'w')
    json.dump(video_metadata, video_metadata_out)

    out.release()
    cap.release()
    cv2.destroyAllWindows()

    print "done annotating"

    return output_name

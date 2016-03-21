import cv2
import numpy as np
import json

def add_side_borders(frame):
    frame = cv2.resize(frame, (480, 360))
    white = np.ones((360,180,3), dtype=np.uint8)
    white = white * 255
    left_image = np.concatenate((white,frame), axis=1)
    layered = np.concatenate((left_image,white), axis=1)
    return layered

def add_labels(layered):

    default_font = cv2.FONT_HERSHEY_SIMPLEX

    # Frame is now 240 rows by 560 columns
    cv2.putText(img = layered, 
                text = "Right Head", 
                org = (2,30), 
                fontFace = default_font, 
                fontScale = 1, 
                color = (0,0,0), 
                thickness = 2)

    cv2.putText(img = layered, 
                text = "Turn", 
                org = (50,60), 
                fontFace = default_font, 
                fontScale = 1, 
                color = (0,0,0),
                thickness = 2)

    cv2.putText(img = layered, 
                text = "Left Head", 
                org = (675,30), 
                fontFace = default_font, 
                fontScale = 1, 
                color = (0,0,0),
                thickness = 2)

    cv2.putText(img = layered, 
                text = "Turn", 
                org = (715,60), 
                fontFace = default_font, 
                fontScale = 1, 
                color = (0,0,0),
                thickness = 2)

    cv2.putText(img = layered, 
                text = "Right Lane", 
                org = (2,200), 
                fontFace = default_font, 
                fontScale = 1, 
                color = (0,0,0),
                thickness = 2)

    cv2.putText(img = layered, 
                text = "Change", 
                org = (30,235), 
                fontFace = default_font, 
                fontScale = 1, 
                color = (0,0,0),
                thickness = 2)

    cv2.putText(img = layered, 
                text = "Left Lane", 
                org = (675,200), 
                fontFace = default_font, 
                fontScale = 1, 
                color = (0,0,0),
                thickness = 2)

    cv2.putText(img = layered, 
                text = "Change", 
                org = (690,235), 
                fontFace = default_font, 
                fontScale = 1, 
                color = (0,0,0),
                thickness = 2)

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
        cv2.rectangle(frame,(180,0),(660,360),sent_to_colour(sentiment),5)
        cv2.putText(img = frame, 
                    text = sent_to_text(sentiment), 
                    org = (20,120), 
                    fontFace = default_font, 
                    fontScale = 1.5, 
                    color = sent_to_colour(sentiment),
                    thickness = 2)

    if (event_type == "right_lane_change"):

        # RIGHT LANE
        cv2.rectangle(frame,(180,0),(660,360),sent_to_colour(sentiment),5)
        cv2.putText(img = frame, 
                    text = sent_to_text(sentiment), 
                    org = (20,290), 
                    fontFace = default_font, 
                    fontScale = 1.5, 
                    color = sent_to_colour(sentiment),
                    thickness = 2)

    if (event_type == "left_turn"):

        # LEFT HEAD
        cv2.rectangle(frame,(180,0),(660,360),sent_to_colour(sentiment),5)
        cv2.putText(img = frame, 
                    text = sent_to_text(sentiment), 
                    org = (700,120), 
                    fontFace = default_font, 
                    fontScale = 1.5, 
                    color = sent_to_colour(sentiment),
                    thickness = 2)

    if (event_type == "left_lane_change"):

        # LEFT LANE
        cv2.rectangle(frame,(180,0),(660,360),sent_to_colour(sentiment),5)
        cv2.putText(img = frame, 
                    text = sent_to_text(sentiment), 
                    org = (700,290), 
                    fontFace = default_font, 
                    fontScale = 1.5, 
                    color = sent_to_colour(sentiment),
                    thickness = 2)

    return frame

def annotate_video(input_name, output_name, head_events_list, lane_events_list, video_metadata_file):
    
    FRAME_RESIZE = (840,360)
    FOURCC = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    FRAME_RATE = 20

    cap = cv2.VideoCapture(input_name)
    out = cv2.VideoWriter(filename = output_name, fourcc = FOURCC, fps = FRAME_RATE, frameSize = FRAME_RESIZE)
    idx = 0

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

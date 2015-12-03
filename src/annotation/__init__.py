import cv2

def annotate_video(video_name, events_list):
    
    FRAME_RESIZE = (320,240)
    FOURCC = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    FRAME_RATE = 50
    OUTPUT_NAME = "annotated.avi"

    print events_list
    num_events = len(events_list)

    if (num_events == 0):
        return video_name

    cap = cv2.VideoCapture(video_name)
    out = cv2.VideoWriter(filename = OUTPUT_NAME, fourcc = FOURCC, fps = FRAME_RATE, frameSize = FRAME_RESIZE)

    # SO UGLY
    event_start_idx = events_list[0][0]
    event_end_idx = events_list[0][1]
    event_idx = 0
    idx = 0

    while (cap.isOpened()):

        (ret, frame) = cap.read()
        if ret==True:
            
            idx += 1
            if (idx > event_start_idx and idx < event_end_idx):
                cv2.rectangle(frame,(20,20),(300,220),(0,255,0), 2)

            if (idx > event_end_idx):    
                event_idx += 1
                if (event_idx >= num_events):
                    break

                # Find new start and end
                event_start_idx = events_list[event_idx][0]
                event_end_idx = events_list[event_idx][1]

            out.write(frame)

        else:
            break
            
    out.release()
    cap.release()
    cv2.destroyAllWindows()

    return OUTPUT_NAME
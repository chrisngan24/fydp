import cv2

def annotate_video(input_name, output_name, events_list, colormap):
    
    FRAME_RESIZE = (320,240)
    FOURCC = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
    FRAME_RATE = 20

    num_events = len(events_list)
    if (num_events == 0):
        return input_name

    cap = cv2.VideoCapture(input_name)
    out = cv2.VideoWriter(filename = output_name, fourcc = FOURCC, fps = FRAME_RATE, frameSize = FRAME_RESIZE)

    # SO UGLY
    event_start_idx = events_list[0][0]
    event_end_idx = events_list[0][1]
    event_type = events_list[0][2]
    event_color = colormap.get(event_type)
    event_idx = 0
    idx = 0

    while (cap.isOpened()):

        (ret, frame) = cap.read()
        if ret==True:
            
            idx += 1
            if (idx > event_start_idx and idx < event_end_idx):
                cv2.rectangle(frame,(20,20),(300,220),event_color,2)

            if (idx > event_end_idx):    
                event_idx += 1
                if (event_idx >= num_events):
                    break

                # Find new start and end
                event_start_idx = events_list[event_idx][0]
                event_end_idx = events_list[event_idx][1]
                event_type = events_list[event_idx][2]
                event_color = colormap.get(event_type)

            out.write(frame)

        else:
            break
            
    out.release()
    cap.release()
    cv2.destroyAllWindows()

    return output_name

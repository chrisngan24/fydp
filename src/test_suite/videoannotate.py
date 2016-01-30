import cv2
import sys

print "#########################################"
print "To Use: python videoplay.py <video_file>"
print "Press l to jump forward a frame"
print "Press k to jump backward a frame"
print "Press p to annotate an event start or end"
print "Press q to quit out"
print "#########################################"

video_name = sys.argv[1]
happy = False
colormap = {'left_head': (0,255,0), 'right_head': (255,0,0), 'left_wheel': (255,255,0), 'right_wheel': (255,0,255)}

############################################
# Firstly load all the frames into data
#############################################

while (not happy):
    
    all_frames = {}
    frame_index = 0
    max_index = 0
    cap = cv2.VideoCapture(video_name)

    while(cap.isOpened()):
        
        (ret, frame) = cap.read()
        if ret==True:
            all_frames[frame_index] = frame
        else:
            max_index = frame_index
            break
        frame_index += 1

    print "Loaded Video into memory. Let's step through"

    ###############################################
    # Step through the video
    ###############################################

    # While within the bounds of the video
    frame_index = 0
    head_event_start = -1
    head_event_end = -1
    wheel_event_start = -1
    wheel_event_end = -1
    events_list = []

    while (frame_index < max_index):

        cv2.imshow('frame', all_frames[frame_index])

        k = cv2.waitKey(0)
        if k == ord('q'):
            break
        elif k == ord('l'):
            frame_index += 1
        elif k == ord('k'):
            if (frame_index == 0):
                print "Already at start of video. Cannot step back."
            else:
                frame_index -= 1
        
        # Annotate an event
        elif k == ord('p'):

            print "What type of annotation is this?"
            print "1 - Left Head Turn"
            print "2 - Right Head Turn"
            print "3 - Left Wheel Turn"
            print "4 - Right Wheel Turn"
            c = cv2.waitKey(0)

            if (c == ord('1')): 
                event_type = "left_head"
            elif (c == ord('2')): 
                event_type = "right_head"
            elif (c == ord('3')): 
                event_type = "left_wheel"
            elif (c == ord('4')): 
                event_type = "right_wheel"
            else:
                event_type = "invalid"
                print "Invalid input, back to main menu."
    
            print "Selected: " + event_type

            # A head turn event
            if (c == ord('1') or c == ord('2')):

                # Starting new event
                if head_event_start < 0:
                    head_event_start = frame_index
                    print "Head turn event start at frame " + str(frame_index)
                # Closing an event
                elif head_event_end < 0:
                    head_event_end = frame_index
                    event = dict(start = head_event_start, end = head_event_end, type = event_type)
                    print event
                    events_list.append(event)
                    head_event_start = -1
                    head_event_end = -1

            # A wheel turn event
            if (c == ord('3') or c == ord('4')):

                # Starting new event
                if wheel_event_start < 0:
                    wheel_event_start = frame_index
                    print "Wheel turn event start at frame " + str(frame_index)
                # Closing an event
                elif wheel_event_end < 0:
                    wheel_event_end = frame_index
                    event = dict(start = wheel_event_start, end = wheel_event_end, type = event_type)
                    print event
                    events_list.append(event)
                    wheel_event_start = -1
                    wheel_event_end = -1

            print "Press l to jump forward a frame"
            print "Press k to jump backward a frame"
            print "Press p to annotate an event start or end"

    print events_list
    print "Done annotating. Let's see your annotated video."

    idx = 0
    event_idx = 0
    num_events = len(events_list)

    event_start_idx = events_list[0]['start']
    event_end_idx = events_list[0]['end']
    event_type = events_list[0]['type']
    event_color = colormap.get(event_type)

    while (idx <= (max_index - 1)):
     
        frame = all_frames[idx]
        idx += 1

        if (idx > event_start_idx and idx < event_end_idx):
            cv2.rectangle(frame,(20,20),(300,220),event_color,2)

        if (idx > event_end_idx):    
            event_idx += 1
            if (event_idx >= num_events):
                event_start_idx = max_index + 1
                event_end_idx = max_index + 2
            else:
                # Find new start and end
                event_start_idx = events_list[event_idx]['start']
                event_end_idx = events_list[event_idx]['end']
                event_type = events_list[event_idx]['type']
                event_color = colormap.get(event_type)

        cv2.imshow("frame", frame)
        cv2.waitKey(20)

    print "Are you happy with this video? (y/n)"
    k = cv2.waitKey(0)

    if k == ord('y'):
        happy = True
        f = open(video_name.split(".")[0] + str(".txt"), 'w')
        f.write(str(events_list))
        print "Bye!"
    else:
        print "Restarting your session..."

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()

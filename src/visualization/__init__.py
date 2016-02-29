import matplotlib.pyplot as plt
from matplotlib import gridspec

import cv2
import sys
import numpy as np

class PointSelector:
    first = True
    
    def __init__(self, line):
        self.line = line
        line.figure.canvas.mpl_connect('pick_event', self)

    def __call__(self, event):
        thisline = event.artist
        xdata = int(event.mouseevent.xdata)
        x_index = np.where(thisline.get_xdata()==xdata)[0]

        ydata = thisline.get_ydata()[x_index]
        ax = thisline.axes
        
        if PointSelector.first == True:
            ax = thisline.axes
            PointSelector.point, = ax.plot(xdata, ydata, 'ro')
            self.line.figure.canvas.draw()
            PointSelector.first = False
        else:
            ax = thisline.axes
            if PointSelector.point.axes == ax:
                PointSelector.point.set_data(xdata, ydata)
                PointSelector.point.figure.canvas.draw()
            else:
                PointSelector.point.remove()
                PointSelector.point, = ax.plot(xdata, ydata, 'ro')
                PointSelector.point.figure.canvas.draw()


class Visualize(object):
    def __init__(self, df, events_hash, video_name, data_direc):
        self.df = df
        self.events = events_hash
        self.data_direc = data_direc
        self.video_name = video_name

        gs = gridspec.GridSpec(2, 1)
        gs.update(hspace=0.5, right=0.8)

        plt.style.use('ggplot')

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(gs[0,:])
        self.ax2 = self.fig.add_subplot(gs[1,:])

    def visualize(self):
        self.make_line_plot(
            self.ax1,
           'timestamp_x',
           ['noseX'],
           title='Position of Nose',
           ylabel='Nose X-coord in the Frame',
           xlabel='# of Samples',
           )

        self.make_line_plot(
            self.ax2,
            'timestamp_x',
            ['theta'],
            title='Angle of Wheel',
            ylabel='Theta (degrees)',
            xlabel='# of Samples',
            )

        self.mark_event(
            self.ax1,
            self.events['head_turns'],
            )

        self.mark_event(
            self.ax2,
            self.events['lane_changes'],
            )

        plt.savefig('%s/fused_plot.png' %self.data_direc)

        plt.show(block=False)

        cap = cv2.VideoCapture(self.video_name)

        all_frames = {}
        frame_index = 0
        max_index = 0
        
        while(cap.isOpened()):
            
            (ret, frame) = cap.read()
            if ret==True:
                all_frames[frame_index] = frame
            else:
                max_index = frame_index
                break
            frame_index += 1

        frame_index = 0

        while (frame_index < max_index):
            
            cv2.imshow('frame', all_frames[frame_index])
            k = cv2.waitKey(0)
            if k == ord('p'):
                frame_index = PointSelector.point.get_xdata()
                y_data = PointSelector.point.get_ydata()
            elif k == ord('l'):
                frame_index += 1
            elif k == ord('k'):
                frame_index -= 1
            elif k == ord('q'):
                break

            PointSelector.point.set_data(frame_index, y_data)
            PointSelector.point.figure.canvas.draw()

        cap.release()
        cv2.destroyAllWindows()


    def make_line_plot(self, ax, x_col, y_col, 
            title='', 
            ylabel='',
            xlabel='',
            ):
        for col in y_col:
            line, = ax.plot(self.df[col], label=col, picker=50)

        e = PointSelector(line)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if len(y_col) > 1:
            ax.legend(bbox_to_anchor=(1.25, 1.05))

    def mark_event(self, ax, events):

        color_sets = ['black', 'magenta', 'yellow', 'green']
        color_index = 0

        for event_name, indices in events.iteritems():
            curr_color = color_sets[color_index]
            for i in indices:
                ax.axvline(x=i, linewidth=1, color=curr_color, linestyle='dashed', label=event_name)
            color_index += 1

        # stop matplotlib from repeating labels in legend for axvline
        handles, labels = ax.get_legend_handles_labels()
        handle_list, label_list = [], []
        for handle, label in zip(handles, labels):
            if label not in label_list:
                handle_list.append(handle)
                label_list.append(label)
        
        ax.legend(handle_list, label_list, fontsize=8, loc='upper right', bbox_to_anchor=(1.28, 1.03))




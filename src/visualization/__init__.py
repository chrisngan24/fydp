import matplotlib.pyplot as plt
from matplotlib import gridspec

import cv2
import sys
import numpy as np

class PointSelector:
    lines = []
    axes = set()
        
    def __init__(self, parent, line):
        self.line = line
        self.parent = parent
        line.figure.canvas.mpl_connect('pick_event', self)
        PointSelector.axes.add(line.axes)

    def __call__(self, event):
        thisline = event.artist
        xdata = int(event.mouseevent.xdata)
        self.parent.show_frame(xdata)

        if len(PointSelector.lines) == 2:
            PointSelector.lines[0].remove()
            PointSelector.lines[1].remove()
            PointSelector.lines = []

        for ax in list(PointSelector.axes):
            l = ax.axvline(x=xdata, color='red', linewidth=1, linestyle='solid')
            PointSelector.lines.append(l)
            l.figure.canvas.draw()


class Visualize(object):
    
    def __init__(self, df, events_hash, video_name, data_direc):
        self.df = df
        self.events = events_hash
        self.data_direc = data_direc
        self.video_name = video_name
        self.all_frames = {}
        self.x_data = 0

        gs = gridspec.GridSpec(2, 1)
        gs.update(hspace=0.5, right=0.8)

        plt.style.use('ggplot')

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(gs[0,:])
        self.ax2 = self.fig.add_subplot(gs[1,:])

    def show_frame(self, x_data):
        
        self.x_data = x_data
        frame_index = self.df['frameIndex'][self.x_data]
        frame2show = self.all_frames[frame_index]
        cv2.imshow("frame", frame2show)
        cv2.waitKey(1)

    def visualize(self, is_interact=True):
        self.make_line_plot(
            self,
            self.ax1,
           'timestamp_x',
           ['noseX'],
           title='Position of Nose',
           ylabel='Nose X-coord in the Frame',
           xlabel='# of Samples',
           )

        self.make_line_plot(
            self,
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
            self.events['head_sentiment']
            )

        self.mark_event(
            self.ax2,
            self.events['lane_changes'],
            self.events['lane_sentiment']
            )

        plt.savefig('%s/fused_plot.png' %self.data_direc)
        
        if not is_interact:
            return

        plt.show(block=False)

        cap = cv2.VideoCapture(self.video_name)

        self.all_frames = {}
        frame_index = 0
        max_index = 0
        
        while(cap.isOpened()):
            
            (ret, frame) = cap.read()
            if ret==True:
                self.all_frames[frame_index] = frame
            else:
                max_index = frame_index
                break
            frame_index += 1

        frame_index = 0

        while True:
            
            if frame_index < 0:
                frame_index = 1

            if frame_index >= max_index:
                frame_index = max_index - 1

            self.show_frame(self.x_data)
            k = cv2.waitKey(0)
            hasPlot = len(PointSelector.lines) == 2
            if k == ord('p'):
                if hasPlot:
                    self.x_data = PointSelector.lines[0].get_xdata()[0]                    
            elif k == ord('l'):
                self.x_data += 2
            elif k == ord('k'):
                self.x_data -= 2
            elif k == ord('q'):
                break

            if hasPlot:
                PointSelector.lines[0].set_xdata([self.x_data, self.x_data])
                PointSelector.lines[1].set_xdata([self.x_data, self.x_data])
                PointSelector.lines[0].figure.canvas.draw()
                PointSelector.lines[1].figure.canvas.draw()

        cap.release()
        cv2.destroyAllWindows()


    def make_line_plot(self, parent, ax, x_col, y_col, 
            title='', 
            ylabel='',
            xlabel='',
            ):
        for col in y_col:
            line, = ax.plot(self.df[col], label=col, picker=50)

        e = PointSelector(parent, line)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if len(y_col) > 1:
            ax.legend(bbox_to_anchor=(1.25, 1.05))

    def mark_event(self, ax, events, sentiments):

        color_sets = ['black', 'magenta', 'yellow', 'green']
        color_index = 0

        for i in xrange(len(events)):
            event = events[i]
            sentiment = sentiments[i][0]
            if sentiment:
                ax.axvspan(event[0], event[1], alpha = 0.25, color = 'g')
            else:
                ax.axvspan(event[0], event[1], alpha = 0.25, color = 'r')

        # stop matplotlib from repeating labels in legend for axvline
        handles, labels = ax.get_legend_handles_labels()
        handle_list, label_list = [], []
        for handle, label in zip(handles, labels):
            if label not in label_list:
                handle_list.append(handle)
                label_list.append(label)
        
        ax.legend(handle_list, label_list, fontsize=8, loc='upper right', bbox_to_anchor=(1.28, 1.03))




from flask import Flask,render_template
app = Flask(__name__, static_url_path='')

import json
import pandas as pd

FPS = 20


def flatten_df(df, meta):
    """
    maps the fused.csv to 
    data type that chart.js 
    will like
    """
    df_og = df.copy()
    df.index = df['frameIndex']
    df = df.reindex(index=list(xrange(0,meta['frames'])), method='nearest')
    vid_length = meta['frames']/float(meta['fps'])
    frames = list(xrange(0, meta['frames']))
    noseX = df['noseX'].tolist()
    theta = df['theta'].tolist()
    headData = []
    for i in xrange(meta['frames']):
       headData.append(dict(
           x = frames[i],
           y = noseX[i],
           ))
    wheelData = []
    for i in xrange(meta['frames']):
       wheelData.append(dict(
           x = frames[i],
           y = theta[i],
           ))
    headEvents = []
    for head_event in meta['head_events']:
        start_frame = df_og.loc[int(head_event[0])]['frameIndex']
        end_frame   = df_og.loc[int(head_event[1])]['frameIndex']
        sentiment   = head_event[2]
        headEvents.append(dict(
            startFrame=start_frame,
            endFrame=end_frame,
            sentimentGood=sentiment,
            ))

    ## COuld be function, but SHIP IT
    laneEvents = []
    for lane_event in meta['lane_events']:
        start_frame = df_og.loc[int(lane_event[0])]['frameIndex']
        end_frame   = df_og.loc[int(lane_event[1])]['frameIndex']
        sentiment   = lane_event[2]
        laneEvents.append(dict(
            startFrame=start_frame,
            endFrame=end_frame,
            sentimentGood=sentiment,
            ))


    return dict(
        frames=frames,
        noseX=noseX,
        headData=headData,
        wheelData=wheelData,
        headEvents=headEvents,
        laneEvents=laneEvents,
        )

def render_interactive(data_dir = 'default'):
    base_dir = 'static'
    m_dir = 'data/%s' % data_dir
    full_dir = base_dir + '/'  + m_dir
    df = pd.read_csv(full_dir + '/fused.csv')
    meta_str = ''.join(open(full_dir + '/annotated_metadata.json', 'r').readlines())
    print meta_str
    meta = json.loads(meta_str)
    frame_per_second = meta['fps']
    frames = meta['frames']

    fused_meta = flatten_df(df, meta)
    video_meta = dict(
            fps=frame_per_second,
            frames=meta['frames'],
            video_file = m_dir + '/annotated_fused.mp4',
            video_time=frames/frame_per_second,
            )
    return render_template('index.html', fused=fused_meta, video=video_meta)


@app.route("/")
def root():
    return render_interactive()
    
if __name__ == "__main__":
    app.run(debug=True)

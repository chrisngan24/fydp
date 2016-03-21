from flask import Flask,render_template
app = Flask(__name__, static_url_path='')

import json
import pandas as pd
import os

FPS = 20


def flatten_df(df, meta):
    """
    maps the fused.csv to 
    data type that chart.js 
    will like
    """
    df_og = df.copy()
    #df.index = df['frameIndex']
    #df = df.groupby('frameIndex', as_index=False).first().reindex(
    #        index=list(xrange(1,meta['frames']+1)), method='backfill')
    video_time = meta['frames']/float(meta['fps'])
    #frames = list(xrange(1, meta['frames']+1))
    frames = df['frameIndex'].tolist()
    noseX = []
    if 'noseX' in df.columns.tolist():
        noseX = df['noseX'].fillna(0).tolist()
    theta = []
    if 'theta' in df.columns.tolist():
        theta = df['theta'].fillna(0).tolist()
    else:
        theta = [0] * len(df)
    headData = []
    for i in xrange(len(frames)):
       headData.append(dict(
           x = frames[i],
           y = noseX[i],
           ))
    wheelData = []
    for i in xrange(len(frames)):
       wheelData.append(dict(
           x = frames[i],
           y = theta[i],
           ))
    headEvents = []
    if meta['head_events'] != None:
        for head_event in meta['head_events']:
            #start_frame = df_og.loc[int(head_event[0])]['frameIndex']
            #end_frame   = df_og.loc[int(head_event[1])]['frameIndex']
            start_frame = int(head_event[0])
            end_frame = int(head_event[1])

            event = head_event[2]
            sentiment   = head_event[3]
            headEvents.append(dict(
                startFrame=start_frame,
                endFrame=end_frame,
                sentimentGood=sentiment,
                ))

    ## COuld be function, but SHIP IT
    laneEvents = []
    lane_sentiment_count = 0.
    counter = 0
    if meta['lane_events'] != None:
        lane_events = meta['lane_events']
        lane_events = sorted(lane_events, key=lambda x: x[0], )
        for lane_event in lane_events:
            #start_frame = df_og.loc[int(lane_event[0])]['frameIndex']
            #end_frame   = df_og.loc[int(lane_event[1])]['frameIndex']
            start_frame = int(lane_event[0])
            end_frame = int(lane_event[1])
            event       = lane_event[2]
            sentiment   = lane_event[3]
            if sentiment:
                sentiment_reason = 'Good job'
            else:
                sentiment_reason = 'Poor form'
            if len(lane_event) == 5:
                sentiment_reason = lane_event[4]

            lane_sentiment_count += sentiment
            laneEvents.append(dict(
                count = counter,
                eventID='#%s' % (counter + 1),
                videoTime = float(start_frame)/meta['frames'] * video_time,
                startFrame=start_frame,
                endFrame=end_frame,
                sentimentGood=sentiment,
                sentimentReason=sentiment_reason
                ))
            counter += 1

    grade = '--'
    overall = ''
    if len(laneEvents) > 0:
        score = lane_sentiment_count / len(laneEvents)
        overall = 'good'
        if score > 0.8:
            grade = 'A'
        elif score > 0.7:
            grade = 'B'
        else:
            grade = 'F'
            overall = 'bad'


    return dict(
        frames=frames,
        noseX=noseX,
        headData=headData,
        wheelData=wheelData,
        headEvents=headEvents,
        laneEvents=laneEvents,
        grade=grade,
        overall=overall,
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
            video_time=float(frames)/frame_per_second,
            session_name=data_dir,
            )
    return render_template('index.html', fused=fused_meta, video=video_meta)


def render_home():
    m_dir = 'data/'
    full_dir = 'static/' + m_dir
    paths = os.listdir(full_dir)
    urls = []
    for path in paths:
        if not path.startswith('.'):
            meta = json.loads(
                    ''.join(open(full_dir + path + '/annotated_metadata.json', 'r').readlines())
                    )
            video_time= '%.2f (s)' % (meta['frames']/ float(meta['fps']))
            url = dict(
                path=m_dir + path,
                url=path,
                title=path,
                video_time=video_time,
                )
                        # so that recent is always at the top
            if path == 'recent':
                pass
            else:
                urls.append(url)
    return render_template('home.html', urls=urls)


###
# App Routing
###
@app.route("/")
def route_home():
    return render_home()
    

@app.route("/report/<data_dir>")
def route_report(data_dir):
    print data_dir
    return render_interactive(data_dir=data_dir)

if __name__ == "__main__":
    app.run(debug=True)

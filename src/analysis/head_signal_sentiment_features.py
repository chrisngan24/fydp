import numpy as np
import pandas as pd

def compute_signal_features(df):
    sampling_rate = (df['time'].max() - df['time'].min()) / len(df)
    df_box = df[df['isFrontFace'] != 0]
    # find the "average" box on the face in the signal
    average_bottom = df_box['faceBottom'].mean()
    average_top = df_box['faceTop'].mean()
    average_left = df_box['faceLeft'].mean()
    average_right = df_box['faceRight'].mean()
    x_dist = df['noseX'].max() - df['noseX'].min(0)
    y_dist = df['noseY'].max() - df['noseY'].min(0)
    speed_norm = df.apply(
        lambda x: np.linalg.norm([
            x['noseX_1_steps'], 
            x['noseY_1_steps'],
        ]), 
        axis=1,
        )
    data_points = float(len(df))
    event_features=dict()
    event_features['signal_length'] = \
        df['time'].max() - df['time'].min()
    event_features['delta_x'] = \
        abs(x_dist / (average_right - average_left))
    event_features['delta_y'] = \
        abs(y_dist / (average_top - average_bottom))

    event_features['front_face_count'] = \
        sum(df['isFrontFace'] == 1) / data_points
    event_features['slowSpeedCount'] = sum(speed_norm < 1)

    if 'turn_sentiment' in df.columns.tolist():
        # for training the model
        event_features['good_turn'] = \
            str(df['turn_sentiment'].max().find('good') != -1)
    return event_features



if __name__ == '__main__':
    df = pd.read_csv('data/look_left_good/events/1453773920-0.csv')
    print compute_signal_features(df)

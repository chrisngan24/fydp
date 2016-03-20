import pandas as pd
import os

data_direc = os.path.join( "data")

if __name__ == "__main__":

    pos_label = 2
    neg_label = 1
    null_label = 0

    left_dfs = []
    right_dfs = []
    neg_dfs = []

    for subdir, dirs, files in os.walk(data_direc):
        for d in dirs:
            df = pd.read_csv("%s/fused.csv" %os.path.join(data_direc, d))
            events = eval(open("%s/events.txt" %os.path.join(data_direc, d), 'r').read())
            if len(events) > 0:
                try:
                    start = events[0][0]
                    end = events[0][1]
                    df.iloc[start:end].reindex().to_csv("%s/model.csv" %os.path.join(data_direc, d))
                except IndexError:
                    continue


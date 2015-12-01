import sys
sys.path.append('../opencv-test')
import os
import time
import pandas as pd 

import klt

if __name__ == '__main__':
    sub_dir = sys.argv[1]
    base_dir = 'data'
    output_dir = '%s/%s' % (base_dir, sub_dir)
    if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    output_file = '%s/%s.csv' % (
            output_dir, 
            str(int(time.time())),
            )
    df = None
    events = []
    try:
        df = klt.run(events)
    finally:
        print events
        if df == None:
            df = pd.DataFrame(events)
        if len(df) > 0:
            print 'end data collection'
            print df.head()
            df.to_csv(output_file, index=False)


        print 'copying data to: ', 
        

import sys
sys.path.append('../opencv-test')
import os
import time
import pandas as pd 

import klt

if __name__ == '__main__':
    sub_dir = sys.argv[1]
    if len(sys.argv) == 3:
        file_name = sys.argv[2]
    else:
        file_name = str(int(time.time()))
    base_dir = 'data'
    output_dir = '%s/%s' % (base_dir, sub_dir)
    if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    output_file = '%s/%s.csv' % (
            output_dir, 
            file_name,
            )
    df = None
    events = []
    try:
        df = klt.run(events)
    finally:
        print events
        if len(df) > 0:
            print 'end data collection'
            print df.head()
            print 'copying data to: ' + file_name
            df.to_csv(output_file, index=False)
        

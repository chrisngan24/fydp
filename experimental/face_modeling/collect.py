import sys
sys.path.append('../opencv-test')
import os
import time

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
    try:
        df = klt.run()
    finally:

        print 'end data collection'
        print df.head()
        df.to_csv(output_file, index=False)


        print 'copying data to: ', 
        

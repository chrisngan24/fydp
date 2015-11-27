import sys
sys.path.append('../opencv-test')
import os

import klt

if __name__ == '__main__':
    sub_dir = sys.argv[1]
    base_dir = 'data'
    output_dir = '%s/%s' % (base_dir, sub_dir)
    if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    df = None
    try:
        df = klt.run()
    finally:

        print 'end data collection'
        print df.head()
        df.to_csv(output_dir)


        print 'copying data to: ', 
        

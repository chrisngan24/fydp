import serial
import time
import pandas as pd
import sys
import datetime

BAUD_RATE = 9600 
PORT = '/dev/cu.usbmodem1421'

def get_current_timestamp():
    return (datetime.datetime.utcnow() - \
                datetime.datetime(1970,1,1)).total_seconds() * 1000


if __name__ == '__main__':
    dir_name = 'data'
    file_name = sys.argv[1]
    csv_file = '%s/%s' % (dir_name, file_name)
    print csv_file
    ser = serial.Serial(
        port=PORT,\
        baudrate=BAUD_RATE)
    values = []
    ser.setDTR(1)
    time.sleep(0.25)
    ser.setDTR(0)
    try:
        while True:
            va = ser.readline()[:-2]
            if len(va) > 0 and len(va.split(',')) == 6:
                values.append(
                        {
                            k[0]: k[1] \
                                    for k in \
                                    map(lambda x: x.split(':'), va.split(',')) \
                                    if len(k) == 2
                        })
                print get_current_timestamp()
                values[len(values)-1]['timestamp'] = get_current_timestamp()
                print values[len(values)-1] 
    except:
        if len(values) > 10:
            print 'saving csv to %s' % csv_file
            df = pd.DataFrame(values)
            df.to_csv(csv_file)


    finally:
        ser.close()


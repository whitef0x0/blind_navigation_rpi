import serial
from start import setup_TFNet

import threading

import logging
logging.basicConfig(filename='./listen_serial.log',level=logging.DEBUG)

#Constants
'''
g_video_state can either be STOPPED or RUNNING
'''

#Shared variables (shared between threads)
g_video_state = "STOPPED" #Initialize to STOPPED

#Mutexes for shared variables
#g_video_state_mutex = threading.Lock()

def serialWaitFor(conn, waitfor_char, receive_char):
    input = conn.read()
    while(input != waitfor_char):
        input = conn.read()
    
    conn.write(receive_char)

class GetSerialThread(threading.Thread):
    def __init__(self, conn):
        threading.Thread.__init__(self)
        self.conn = conn

    def run(self):
        print("GetSerialThread Started")

        global g_video_state
        while(True):
            print('Waiting for de1 start video command')
            serialWaitFor(self.conn, b'V', b'v')
            print('Received de1 start video command')
            g_video_state = "RUNNING"

            #Wait until video has stopped before sending stop command to DE-1
            print('Waiting for de1 stop video command')
            serialWaitFor(self.conn, b'S', b's')
            print('Received de1 stop video command')
            g_video_state = "STOPPED"

def main():

    conn = serial.Serial("/dev/cu.usbserial", 115200, timeout=1)

    print("start serial conn with de1")
    tfnet = setup_TFNet()

    d1 = GetSerialThread(conn)
    d1.setDaemon(True)

    global g_video_state
    g_video_state = "STOPPED"
    d1.start()


    while(True):
        while(g_video_state != "RUNNING"): continue;
        camera = tfnet.setup_camera()
        print('Setup darkflow camera')

        should_break = False
        while(g_video_state != "STOPPED" and should_break == False):
            should_break = tfnet.process_frame()
        print('Captured all frames')
        should_break = False

        tfnet.teardown_camera()
        print('Saved video and metadata to cloud')

if __name__ == '__main__':
    main()

        
        

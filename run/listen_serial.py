import serial
from start import setup_TFNet

import threading
import soundvision

#Constants
#   g_current_state can either be STOPPED or RUNNING
#   g_current_category can either be NONE, TRACKING or SOUNDVISION

#Shared variables (shared between threads)
g_current_state = "STOPPED"

g_current_category = "NONE"

#Mutexes for shared variables
g_serial_state_mutex = threading.Lock()

def serialWaitFor(conn, waitfor_chars):
    input = conn.read()
    while(input not in waitfor_char):
        input = conn.read()
    
    receive_char = input.lower()
    conn.write(receive_char)
    return input

def waitForStop(conn):
    serialWaitFor(conn, ['S'])

class GetSerialThread(threading.Thread):
    def __init__(self, conn):
        threading.Thread.__init__(self)
        self.conn = conn

    def run(self):
        print("GetSerialThread Started")

        global g_video_state
        while(True):
            print('Waiting for de1 start video command')
            commandType = serialWaitFor(self.conn, ['V', 'H'])
            g_current_state = "RUNNING"

            if commandType == 'V':
                g_current_category = "TRACKING"
                print('Received de1 start video command')

                #Wait until video has stopped before sending stop command to DE-1
                print('Waiting for de1 stop video command')
                waitForStop(self.conn)
                print('Received de1 stop video command')
            else:
                g_current_category = "SOUNDVISION"
                print('Received de1 start soundvision command')

                #Wait until video has stopped before sending stop command to DE-1
                print('Waiting for de1 stop soundvision command')
                waitForStop(self.conn)
                print('Received de1 stop soundvision command')
            g_current_category = "NONE"
            g_current_state = "STOPPED"
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
        while(g_current_state != "STOPPED" or g_current_category != "NONE"): continue;
        if g_current_category == "TRACKING":
            camera = tfnet.setup_camera()
            print('Setup darkflow camera')

            should_break = False
            while(g_current_state != "STOPPED" and should_break == False):
                should_break = tfnet.process_frame()
            print('Captured all frames for object tracking')
            should_break = False

            tfnet.teardown_camera()
            print('Saved video and metadata to cloud')
        elif g_current_category == "SOUNDVISION":
            camera = soundvision.setup()
            print('Setup soundvision')

            should_break = False
            while(g_current_state != "STOPPED" and should_break == False):
                should_break = soundvision.process_frame(camera)
            print('Captured all frames for soundvision')
            should_break = False

            soundvision.teardown(camera)
            print('Successfully exited soundvision')

if __name__ == '__main__':
    main()

        
        

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

# Wait for connection from out serial ports
def serialWaitFor(conn, waitfor_chars):
    input = conn.read()
    while(str(input, 'ascii') not in waitfor_chars):
        input = conn.read()
    
    receive_char = str(input, 'ascii').lower()
    conn.write(str.encode(receive_char))
    return input

def waitForStop(conn):
    serialWaitFor(conn, ['S'])

# Thread object to start a video with command from DE1-SoC
class GetSerialThread(threading.Thread):
    def __init__(self, conn):
        threading.Thread.__init__(self)
        self.conn = conn

    def run(self):
        print("GetSerialThread Started")

        global g_video_state
        while(True):
            # Connect to DE1 to wait for commands from videos
            print('Waiting for de1 start command')
            commandType = serialWaitFor(self.conn, ['V', 'H'])
            g_current_state = "RUNNING"

            # Switch between video tracking and sound image processing
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

    # Start connection with a external serial device
    conn = serial.Serial("/dev/cu.usbserial", 115200, timeout=1)

    print("start serial conn with de1")
    tfnet = setup_TFNet()

    d1 = GetSerialThread(conn)
    d1.setDaemon(True)

    global g_video_state
    g_video_state = "STOPPED"
    d1.start()

    while(True):
        while(g_current_state == "STOPPED" and g_current_category == "NONE"): continue;
        # Part to set up an object tracking
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
        # Part to set up image sound conversion
        else:
            camera = soundvision.setup()
            print('Setup soundvision')

            if camera is None:
                print("ERROR: Setup failed!")
                raise

            final_byte_array = []
            while(g_current_state != "STOPPED"):
                byte_array, should_break = soundvision.process_frame(camera)
                if should_break is True:
                    final_byte_array.extend(byte_array)
            print('Captured all frames for soundvision')
            should_break = False

            soundvision.teardown(camera, final_byte_array)
            print('Successfully exited soundvision')

if __name__ == '__main__':
    main()

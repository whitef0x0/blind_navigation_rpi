import serial
from start import setup_TFNet
import soundvision

import threading, copy
import cv2
from queue import Queue
from google_speech import Speech
import soundvision_c

class RemoveOldDataQueue(Queue):
    def put(self,*args,**kwargs):
        if self.full():
            try:
                oldest_data = self.get()
                #print('[WARNING]: throwing away old data:'+repr(oldest_data))
            # a True value from `full()` does not guarantee
            # that anything remains in the queue when `get()` is called
            except Queue.Empty:
                pass
        Queue.put(self,*args,**kwargs)

def speakText(text):
    global g_lang, g_sox_effects
    speech = Speech(text, g_lang)
    speech.play(g_sox_effects)

def setup():
    #Initialize all global constants
    #   g_current_state can either be LIVE_VIEW or SAVE_VIDEO
    #   g_current_category can either be TRACKING or SOUNDVISION
    globals()['g_current_state'] = "LIVE_VIEW"
    globals()['g_current_category'] = "SOUNDVISION"
    globals()['g_last_ten_seconds'] = RemoveOldDataQueue(maxsize=30)
    globals()['g_camera'] = getCamera()
    globals()['g_lang'] = "en"
    globals()['g_sox_effects'] = ("speed", "1.5")

    #Initialize Mutexes
    globals()['g_current_state_mutex'] = threading.Lock()
    globals()['g_current_category_mutex'] = threading.Lock()

    #Setup SoundVision
    soundvision.setup()

    #Setup ObjectDetection
    globals()['g_tfnet'] = setup_TFNet()
    globals()['g_tfnet'].setup_camera(globals()['g_camera'])


def getCamera():
    camera = cv2.VideoCapture(0)
    cam_h2w = 720/1280
    expected_width = 640
    expected_height = expected_width * cam_h2w
    camera.set(3,expected_height)
    camera.set(4,expected_width)

    if not camera.isOpened():
        speakText("ERROR: Cannot connect to camera")
        exit("Cannot connect to camera")

    frame = camera.read()  # Dummy read needed with some devices
    return camera

def getCurrentFrame(camera):
    ret, frame = camera.read()

    err = False
    if ret is False or frame is None:
        err = True

    return err, frame

def serialWaitFor(conn, waitfor_chars):
    input = conn.read()
    while(str(input, "ascii") not in waitfor_chars):
        input = conn.read()
    
    receive_char = str(input, "ascii").lower()
    conn.write(str.encode(receive_char))
    return str(input, "ascii")

class PollSerialThread(threading.Thread):
    def __init__(self, conn):
        threading.Thread.__init__(self)
        self.conn = conn

    def run(self):
        print("GetSerialThread Started")

        global g_current_state, g_current_state_mutex, g_current_category, g_current_category_mutex
        while(True):
            print("Waiting for de1 start command")
            commandType = serialWaitFor(self.conn, ["V", "H", "C"])
            g_current_state = "RUNNING"

            if commandType == "H":
                print("Started to switch to sound vision for livestream")
                speakText("Started to switch to sound vision for livestream")
                with g_current_category_mutex:
                    g_current_category = "SOUNDVISION"
                with g_current_state_mutex:
                    g_current_state = "LIVE_VIEW"
            elif commandType == "V":
                print("Started to switch to object tracking for livestream")
                speakText("Started to switch to object tracking for livestream")
                with g_current_category_mutex:
                    g_current_category = "TRACKING"
                with g_current_state_mutex:
                    g_current_state = "LIVE_VIEW"
            elif commandType == "C":
                speakText("Started to save last ten seconds of video")
                with g_current_state_mutex:
                    g_current_state = "SAVE_VIDEO"
            else:
                print("ERROR: Unkown command " + str(commandType))

class SaveVideoThread(threading.Thread):
    def __init__(self, video_filename):
        threading.Thread.__init__(self)
        self.video_filename = video_filename

    def saveFramesAsVideo(self):
        videoWriter = cv2.videoWriter(self.video_filename, -1, 5, (640, 480))
        frames_list = list(self.frames_queue)

        for frame in frames_list:
            videoWriter.write(frame)
        videoWriter.release()

        try:
            file = open(self.video_filename, 'rb')
        except OSError as err:
            print("Could not open file (OSError): {0}".format(err))
            return True, None
        except:
            print("Unexpected error:", sys.exc_info()[0])
            return True, None
        else:
            return False, file

    def uploadVideo(self, file):
        url = 'http://ec2-18-191-1-128.us-east-2.compute.amazonaws.com/video_stream/'
        
        try:
            files = {'file': open(self.video_filename, 'rb')}
        except OSError as err:
            print("Could not open videofile (OSError): {0}".format(err))
            return True
        except:
            print("Could not open videofile (Unexpected error):", sys.exc_info()[0])
            return True

        try:
            r = requests.post(url, files=files)
        except requests.exceptions.RequestException as e:
            print("Could not upload videofile to server: {0}".format(err))
            return True

        try:
            os.remove(self.video_filename)
        except OSError as err:
            print("Could not delete local copy of videofile (OSError): {0}".format(err))
            return True
        except:
            print("Could not delete local copy of videofile (Unexpected error):", sys.exc_info()[0])
            return True

        return False

    def run(self):
        print("SaveVideoThread Started")
        global g_current_state, g_current_state_mutex, g_last_ten_seconds

        self.frames_queue = copy.copy(g_last_ten_seconds)

        while(True):
            while(g_current_state != "SAVE_VIDEO"): continue;
            speakText("Started to save last ten seconds of video")

            err, file = self.saveFramesAsVideo(g_last_ten_seconds)
            if(err):
                speakText("ERROR: Could not save video")
                exit("Could not save video")

            err = self.uploadVideo(file)
            if(err):
                speakText("ERROR: Could not save video")
                exit("Could not save video")

            speakText("Video successfully saved and uploaded")
            with g_current_state_mutex:
                g_current_stat = "LIVE_VIEW"

def mainThread():
    curr_frame = None
    previous_frame = None

    while(True):
        global g_current_state, g_current_state_mutex, g_current_category, g_current_category_mutex, g_camera

        #Get Current Frame from Camera
        previous_frame = curr_frame
        has_error, curr_frame = getCurrentFrame(g_camera)
        if has_error is True:
            speakText("Error: Camera could not be read. Exiting now")
            exit("Camera could not be read")

        #Only save frame to queue when we aren't currently uploading the queue's frames
        global g_last_ten_seconds
        g_last_ten_seconds.put(curr_frame, False)

        if g_current_state == "LIVE_VIEW":
            if g_current_category == "TRACKING":
                global g_tfnet
                has_error = g_tfnet.process_frame(curr_frame, previous_frame)
                if has_error is True:
                    speakText("Error: Object detection failed on current frame. Exiting now")
                    exit("Object Detection failed on current frame")
            elif g_current_category == "SOUNDVISION":
                soundvision_c.process_frame(curr_frame)
            else: 
                with g_current_category_mutex:
                    g_current_category = "TRACKING"
        elif g_current_state != "SAVE_VIDEO":
            with g_current_state_mutex:
                g_current_state = "LIVE_VIEW"

def listen_to_serial_port(is_test=False, test_conn=None):
    conn = None
    if test_conn is not None:
        conn = test_conn
    else:
        conn = serial.Serial("/dev/ttys004", 115200, timeout=1)
        #conn = serial.Serial("/dev/cu.usbserial", 115200, timeout=1)
    default_video_name = "output_video.mov"

    #Initialize global vars and setup soundvision and object tracking
    setup()

    print("start serial conn with de1")

    serial_thread = PollSerialThread(conn)
    serial_thread.setDaemon(True)
    serial_thread.start()

    video_save_thread = SaveVideoThread(default_video_name)
    video_save_thread.setDaemon(True)
    video_save_thread.start()

    if is_test is True:
        sensors_thread = threading.Thread(target=mainThread)
        sensors_thread.setDaemon(True)
        sensors_thread.start()
    else:
        mainThread()

    cv2.namedWindow('LiveFeed', cv.WINDOW_AUTOSIZE);
    cv2.namedWindow('LiveFeed', cv.WINDOW_AUTOSIZE);

if __name__ == "__main__":
    listen_to_serial_port()

        
        

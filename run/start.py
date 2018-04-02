from darkflow.defaults import argHandler #Import the default arguments
import os

FLAGS = argHandler()
FLAGS.setDefaults()

FLAGS.config = "./darkflow/cfg/"
FLAGS.demo = "camera" # video file to use, or if camera just put "camera"
FLAGS.model = "./darkflow/cfg/tiny-yolo-voc.cfg" # tensorflow model
FLAGS.load = "./darkflow/bin/tiny-yolo-voc.weights" # tensorflow weights
# FLAGS.pbLoad = "tiny-yolo-voc-traffic.pb" # tensorflow model
# FLAGS.metaLoad = "tiny-yolo-voc-traffic.meta" # tensorflow weights
FLAGS.threshold = 0.5 # threshold of detection confidance (detection if confidance > threshold )
FLAGS.gpu = 0 #how much of the GPU to use (between 0 and 1) 0 means use cpu
FLAGS.track = True # wheither to activate tracking or not
FLAGS.trackObj = ["aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"]
FLAGS.saveVideo = False  #whether to save the video or not

FLAGS.speech = True #whether to enable text to speech of labels

FLAGS.grayscale = False #whether to read images from webcam as black and white

FLAGS.upload = False #whether or not to upload video to AWS
FLAGS.BK_MOG = False # activate background substraction using cv2 MOG substraction,
                        #to help in worst case scenarion when YOLO cannor predict(able to detect movement, it's not ideal but well)
                        # helps only when number of detection < 3, as it is still better than no detection.
FLAGS.tracker = "sort" # wich algorithm to use for tracking deep_sort/sort (NOTE : deep_sort only trained for people detection )
FLAGS.skip = 0 # how many frames to skip between each detection to speed up the network
FLAGS.csv = False #whether to write csv file or not(only when tracking is set to True)
FLAGS.display = True # display the tracking or not
FLAGS.face_recognition = True # enable face recognition

def start_camera():
    from darkflow.net.build import TFNet
    tfnet = TFNet(FLAGS)
    tfnet.camera()
    exit('Demo stopped, exit.')

if __name__ == "__main__":
    start_camera()

import cv2
import os
import numpy as np

def detect_face(img):
    #convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('./cv2_data/haar_cascades/haarcascade_frontalface_default.xml')

     
    #let's detect multiscale images(some images may be closer to camera than others)
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
     
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
     
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
     
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list 
#of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
     
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
     
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
     
    #let's go through each directory and read images within it
    for dir_name in dirs:
     
        #our subject directories start with letter 's' so
        #ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;
         
        #------STEP-2--------
        #extract label number of subject from dir_name
        #format of dir name = slabel
        #, so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))
        print(label)
         
        #build path of directory containing images for current subject subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
         
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
         
        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:
         
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
             
            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name
             
            #read image
            image = cv2.imread(image_path)
             
            #display an image window to show the image 
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
             
            #detect face
            face, rect = detect_face(image)
             
            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
     
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
     
    return faces, labels

def main():
    print("Preparing data...")
    faces, labels = prepare_training_data("./cv2_data/face_recog/training_data")
    print("Data prepared")
     
    #print total faces and labels
    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.write("./cv2_data/face_recog/david_model.yaml")

if __name__ == '__main__':
    main()
import cv2
import imutils

face_cascade = cv2.CascadeClassifier('./cv2_data/haar_cascades/haarcascade_frontalface_default.xml')

def detect_face(frame):
    """
    detect human faces in image using haar-cascade
    Args:
        frame:
    Returns:
    coordinates of detected faces
    """
    faces = face_cascade.detectMultiScale(frame, 1.3, 2, 0, (20, 20) )
    return faces

def recognize_face(frame_orginal, faces):
    """
    recognize human faces using LBPH features
    Args:
        frame_orginal:
        faces:
    Returns:
        label of predicted person
    """
    predict_label = []
    predict_conf = []
    for x, y, w, h in faces:
        frame_orginal_grayscale = cv2.cvtColor(frame_orginal[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY)
        cv2.imshow("cropped", frame_orginal_grayscale)
        predict_tuple = recognizer.predict(frame_orginal_grayscale)
        a, b = predict_tuple
        predict_label.append(a)
        predict_conf.append(b)
        print("Predition label, confidence: " + str(predict_tuple))
    return predict_label

def draw_faces(frame, faces):
    """
    draw rectangle around detected faces
    Args:
        frame:
        faces:
    Returns:
    face drawn processed frame
    """
    for (x, y, w, h) in faces:
        xA = x
        yA = y
        xB = x + w
        yB = y + h
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    return frame

def save_faces(frame, faces, idx):
    """
    save faces as individual images
    Args:
        frame:
        faces:
    Returns:
    face drawn processed frame
    """
    i = idx
    for (x, y, w, h) in faces:

        if w > 150 and h < 400:
            crop_img = frame[y:y+h, x:x+w]
            cv2.imwrite("./cv2_data/face_recog/training_data/s1/img" + str(i) + ".jpg", crop_img)
            i += 1
    return i

def put_label_on_face(frame, faces, labels):
    """
    draw label on faces
    Args:
        frame:
        faces:
        labels:
    Returns:
        processed frame
    """
    i = 0
    for x, y, w, h in faces:
        cv2.putText(frame, str(labels[i]), (x, y), font, 1, (255, 255, 255), 2)
        i += 1
    return frame

def background_subtraction(previous_frame, frame_resized_grayscale, min_area):
    """
    This function returns 1 for the frames in which the area 
    after subtraction with previous frame is greater than minimum area
    defined. 
    Thus expensive computation of human detection face detection 
    and face recognition is not done on all the frames.
    Only the frames undergoing significant amount of change (which is controlled min_area)
    are processed for detection and recognition.
    """
    frameDelta = cv2.absdiff(previous_frame, frame_resized_grayscale)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    im2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    temp=0
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) > min_area:
            temp=1
    return temp  

def run_camera():   
    camera = cv2.VideoCapture(0)

    cv2.namedWindow('', 0)
    _, frame = camera.read()
    height, width, _ = frame.shape
    cv2.resizeWindow('', width, height)
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    i = 0
    while camera.isOpened():
        _, current_frame = camera.read()

        if current_frame is None:
            print ('\nEnd of Video')
            break

        previous_frame = frame_grayscale
        frame_grayscale = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        out_frame = current_frame

        #Face Detection Code
        min_area=(3000/800)*current_frame.shape[1] 
        frame_grayscale = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        temp = background_subtraction(previous_frame, frame_grayscale, min_area)
        if temp==1:     
            faces = detect_face(frame_grayscale)
            if len(faces) > 0:
                i = save_faces(current_frame, faces, i) + 1
                frame_face_processed = draw_faces(out_frame, faces)

        cv2.imshow('', out_frame)
        choice = cv2.waitKey(1)
        if choice == 27:
            break

if __name__ == '__main__':
    run_camera()
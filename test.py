import numpy as np
import cv2 as cv
import mediapipe as mp
import time
#from mediapipe.tasks.vision import ImageProcessingOptions


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "hand_landmarker.task"
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands = 2
)


model = "face_detection_yunet_2023mar.onnx"
input_size = (640,480)
obj_dect = 	cv.FaceDetectorYN.create(
    model = model, 
    config = "", 
    input_size = input_size,
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=500,
    backend_id=cv.dnn.DNN_BACKEND_OPENCV,
    target_id=cv.dnn.DNN_TARGET_CPU
    ) 



cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

fps = cap.get(cv.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # Default fallback

frame_interval_ms = int(1000 / fps)


HAND_CONNECTIONS = {
    (0,1),(0,5),(0,9),(0,13),(0,17),(1,2),(2,3),(3,4), 
    (5,6), (6,7), (7,8), (5,9),(9,13),(13,17),(2,5),
    (9,10),(10,11),(11,12),
    (13,14), (14,15), (15,16),
    (17,18),(18,19),(19,20)
}

with HandLandmarker.create_from_options(options) as landmarker:
    timestamp = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        
        #Face Detection
        resized_frame = cv.resize(frame, (640,480))
        faces = obj_dect.detect(resized_frame)[1]
        
        if faces is not None:
            for face in faces:
                x,y,w,h = map(int, face[:4])
                cv.rectangle(frame,(x,y), (x+w,y+h), (0,255,0), 2)
                landmarks = face[4:14].reshape(5,2).astype(int)

                for (lx,ly) in landmarks:
                    cv.circle(frame, (lx,ly), 2, (0,0,255), -1)
                
                labels = ["Right Eye", "Left Eye", "Nose", "Right Lip", "Left Lip"]
                for i, (lx,ly) in enumerate(landmarks):
                    cv.putText(frame, labels[i], (lx+5,ly+5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                cv.putText(frame, "Face Score: " + f"{face[-1]:.3f}", (x,y+10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1)
        

        #Hand Detection
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_frame)

        result = landmarker.detect_for_video(mp_image, timestamp)
        timestamp+=frame_interval_ms

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                for landmark_idx, landmark in enumerate(hand):
                    h,w, _ = frame.shape
                    x = int(landmark.x *w)
                    y = int(landmark.y*h)
                    cv.circle(frame, (x,y), 2, (0,255,0), -1)
                    cv.putText(frame, str(landmark_idx), (x+5,y+5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1)
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start_landmark, end_landmark = hand[start_idx], hand[end_idx]
                    h,w, _ = frame.shape
                    start_x, end_x = int(start_landmark.x *w), int(end_landmark.x *w)
                    start_y, end_y = int(start_landmark.y *h), int(end_landmark.y *h)
                    cv.line(frame, (start_x,start_y), (end_x, end_y), (255,0,0), 2)

        

        cv.imshow('Face and Hand Detection', frame)
        if cv.waitKey(1) == ord('q'):
            break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()


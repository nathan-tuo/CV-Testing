import numpy as np
import cv2 as cv
import mediapipe as mp
import time


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


'''def fist_confidence(hand_landmarks, frame_shape):
    h,w = frame_shape
    fingertip_indices = [4,8,12,16,20]
    first_knuckle_indices = [3,7,11,15,20]
    hand_base_indices = [2,5,9,13,17]

    fingertips = [hand_landmarks[i] for i in fingertip_indices]
    hand_base = [hand_landmarks[i] for i in hand_base_indices]
    first_knuckle = [hand_landmarks[i] for i in  first_knuckle_indices]

    distance = 0
    distances = 0
    max_possible_distance = 0
    for i in range(len(fingertip_indices)):
        x_distance = fingertips[i].x *w - hand_base[i].x *w
        y_distance = fingertips[i].y * h - hand_base[i].y * h
        x_1_distance = first_knuckle[i].x *w - hand_base[i].x *w
        y_1_distance = first_knuckle[i].y *h - hand_base[i].y *h
        distance = np.sqrt(np.square(x_distance) + np.square(y_distance))
        distance_1 = np.sqrt(np.square(x_1_distance) + np.square(y_1_distance))
        distances+= (distance + distance_1)/2
        palm_diagonal = np.sqrt(w**2 + h**2) * 0.08  # Rough estimate
        max_possible_distance += palm_diagonal
    
    # Normalize to 0-1 range (lower distance = more closed fist)
    fist_confidence = 1 - (distances / max_possible_distance)
    print(fist_confidence)
    return max(0, min(1, fist_confidence))  # Clamp to 0-1'''

def vector_between(p1,p2):
    return ([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p2[2]])
def angle_between_vectors(v1,v2):
    dp = np.dot(v1,v2)
    norm1 = np.sqrt(np.sum(np.square(v1)))
    norm2 = np.sqrt(np.sum(np.square(v2)))
    return 180 - np.rad2deg(np.arccos(dp/(norm1*norm2)))
def identify_fist_with_angles(hand_landmarks, frame_shape):
    hand_base_indices = [2,5,9,13,17]
    first_knuckle_indices = [3,6,10,14,19]
    middle_knuckle_indices = [4,7,11,15,20] #except thumb (technically fingertip)

    h,w = frame_shape

    hand_base_landmarks = [hand_landmarks[i] for i in hand_base_indices]
    first_knuckle_landmarks = [hand_landmarks[i] for i in first_knuckle_indices]
    middle_knuckle_landmarks = [hand_landmarks[i] for i in middle_knuckle_indices]

    adjusted_hand_base_landmarks = [[lm.x *w, lm.y *h, lm.z] for lm in hand_base_landmarks]
    adjusted_first_knuckle_landmarks = [[lm.x*w, lm.y * h, lm.z] for lm in first_knuckle_landmarks]
    adjusted_middle_knuckle_landmarks = [[lm.x*w, lm.y*h,lm.z] for lm in middle_knuckle_landmarks]

    print(f"{adjusted_hand_base_landmarks} + \n + {adjusted_first_knuckle_landmarks} + \n + {adjusted_middle_knuckle_landmarks}")
    vec1 = [vector_between(adjusted_hand_base_landmarks[i], adjusted_first_knuckle_landmarks[i]) for i in range(len(adjusted_hand_base_landmarks))]
    vec2 = [vector_between(adjusted_first_knuckle_landmarks[i], adjusted_middle_knuckle_landmarks[i]) for i in range(len(adjusted_first_knuckle_landmarks))]
    print(f"vector1: {vec1}, vector2: {vec2}")
    angles_between = [angle_between_vectors(vec1[i],vec2[i]) for i in range(len(vec1))]
    print(f"angles_between: {angles_between} ")
    print(f"Debug - Angle: {np.mean(angles_between)}")
    return np.mean(angles_between)
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

                #confidence = fist_confidence(hand, frame.shape[:2])
                #if(confidence > 0.4):
                    #cv.putText(frame, "Fist", (50,50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                angle = identify_fist_with_angles(hand, frame.shape[:2])
                if(angle <90):
                    cv.putText(frame, "Fist", (50,50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start_landmark, end_landmark = hand[start_idx], hand[end_idx]
                    h,w, _ = frame.shape
                    start_x, end_x = int(start_landmark.x *w), int(end_landmark.x *w)
                    start_y, end_y = int(start_landmark.y *h), int(end_landmark.y *h)
                    cv.line(frame, (start_x,start_y), (end_x, end_y), (255,0,0), 2)

        
        print(f"Frame Shape {frame.shape}")
        cv.imshow('Face and Hand Detection', frame)
        if cv.waitKey(1) == ord('q'):
            break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

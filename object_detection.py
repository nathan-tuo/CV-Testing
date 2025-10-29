import numpy as np
import mediapipe as mp
import cv2 as cv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
#DetectionResult = mp.tasks.components.containers.detections.DetectionResult
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode
model = "efficientdet_lite2.tflite"

options = ObjectDetectorOptions(
    base_options = BaseOptions(model_asset_path = model),
    max_results = 5,
    running_mode = VisionRunningMode.VIDEO,
    score_threshold = 0.5
)

detector = vision.ObjectDetector.create_from_options(options)
cap = cv.VideoCapture(0)

fps = cap.get(cv.CAP_PROP_FPS)
frame_duration_ms = int(1000 / fps) if fps > 0 else 30

if not cap.isOpened():
    print("Webcam not working.")
    exit()

frame_timestamp_ms = 0

while True:
    ret,frame = cap.read()
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)


    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_frame)
    objects = detector.detect_for_video(mp_image, frame_timestamp_ms)

    if objects.detections is not None:
        for detection in objects.detections: 
            start_x, start_y = detection.bounding_box.origin_x, detection.bounding_box.origin_y
            width, height = detection.bounding_box.width, detection.bounding_box.height
            cv.rectangle(frame, (start_x,start_y), (start_x+width, start_y+height), (255,0,0),2)

            category = detection.categories[0]
            category_name = category.category_name
            score = round(category.score,2)

            cv.putText(frame, f"{category_name}: {score}", (start_x,start_y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            
    
    cv.imshow("Frame", frame)
    frame_timestamp_ms+=frame_duration_ms
    if(cv.waitKey(1) == ord("q")):
        break

cap.release()
cv.destroyAllWindows()
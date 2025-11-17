from ultralytics import YOLO
from ultralytics.engine.results import Keypoints
import cv2
# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model


# Load a video (e.g., Jiu Jitsu sparring)
cap = cv2.VideoCapture("C:/Projects/AI/Image_class_bjj/Singleleg/frame12.jpg")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # if num_frames % 5 == 0:
        
    #     cv2.imwrite("C:/Projects/AI/Image_class_bjj/Singleleg/frame%d.jpg" % counter, frame)
    #     counter += 1
    # if not ret:
    #     print("Can't receive frame (stream end?). Exiting ...")
    #     break
    # num_frames += 1
    # Run pose estimation
    results = model(frame)

    # Draw keypoints and boxes on the frame
    annotated_frame = results[0].plot()

    keypoints = results[0].keypoints

    xy = keypoints.xy

    print(xy.shape)

    print(xy[0])
    # for person in results[0].keypoints:
    # person.xy: (num_keypoints, 2)
    # person.conf: (num_keypoints,)
        # print(person.xy)
    

    
    # cv2.imshow("YOLOv8 Pose", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
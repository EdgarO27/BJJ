# import cv2
# from ultralytics import YOLO
# import os
# import csv
# import numpy as np
# from collections import defaultdict
# output_file = "Pose_points.csv"



# parts = ['Nose', 'LeftEye','RightEye','LeftEar','RightEar','LeftShoulder','RightShoulder','LeftElbow','RightElbow','LeftWrist','Right Wrist','LeftHip','RightHip','LeftKnee','RightKnee','LeftAnkle','RightAnkle']

# # if not os.path.exists(output_file):
# #     with open(output_file, "w", newline="") as f:
# #         writer = csv.writer(f)
# #         header = ["frame"]
# #           # 2 people
# #         for i in range(len(parts)):  # 17 keypoints
# #                 header += [f"{parts[i]}"]
# #         writer.writerow(header)
# cap = cv2.VideoCapture(r'C:\Projects\AI\Image_class_bjj\clips\SingleLeg\single_leg_06.mp4')
# model = YOLO('yolov8n-pose.pt')

# frame_count = 0
# pose_points = []
# while True:

#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.resize(frame, (640, 640))

#     # Run pose model
#     results = model.track(frame, persist=True)

#     # YOLO returns a list; take first result
#     result = results[0]
#     track_history = defaultdict(lambda: [])
#     # Count the number of people (pose detections)
#     if result.keypoints is not None:
#         num_people = len(result.keypoints)

#         if num_people == 2:
#             print("People detected:", num_people)
#             for person_id, person_kpts in enumerate(result.keypoints.xyn.numpy().astype(np.uint8)):
#                 print(f"--- Person {person_id} ---")
#                 # pose_points.append((person_id,person_kpts))
#                 # flat = person_kpts.flatten()
#                 # print(person_id)
#                 # print(person_kpts)
#                 # print(flat)
#                 row = [person_id] + person_kpts.tolist() 
#                 with open(output_file, "a", newline="") as f:
#                     writer = csv.writer(f)
#                     writer.writerow(row)
#         frame_count += 1
#     else:
#         num_people = 0

#     # print("People detected:", num_people)
    
    
        

        
#         # for kp_idx, kp in enumerate(person_kpts):
#         #     x = kp
#         #     print(x)
#     # ---- SAVE FRAME ONLY IF 2 PEOPLE DETECTED ----
#     # if num_people == 2:
#     #     annotated = result.plot()
#     #     cv2.imwrite("annon/frame%d.jpg" % frame_count, annotated)
#     #     print(f"Saved Pose/annon/frame_{frame_count:05}.jpg")
    
    
    
#     # for result in results:
#     #     result.save_txt("annon/output.txt")

#     # OPTIONAL: Display window (can remove this)
#     # cv2.imshow("Pose", result.plot())
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
from ultralytics import YOLO
import os
from ultralytics import YOLO
import pathlib
import pandas as pd






model = YOLO('yolo11n-pose.pt')
frame_folder = r"C:\Projects\AI\Image_class_bjj\Pose\Preprocess\Single_leg_frames\2"
pose_store = r"C:\Projects\AI\Image_class_bjj\Pose\Preprocess\Single_Leg_pose\2"
parts = ['Nose', 'LeftEye','RightEye','LeftEar',
         'RightEar','LeftShoulder','RightShoulder','LeftElbow',
         'RightElbow','LeftWrist','Right Wrist','LeftHip','RightHip',
         'LeftKnee','RightKnee','LeftAnkle','RightAnkle']
single_leg_frame = [] # storing in single leg frame folder
single_leg_pose = []  # storing in single leg pose estimation folder
frame_count = 0
folder_count = 2 # folder count that will be used for imwrite for frames
# df = pd.DataFrame(columns = ["Action", "FrameId"])
#Two for loops that will extract folder names from single clips and not single clips

    
cap = cv2.VideoCapture(r"C:\Projects\AI\Image_class_bjj\clips\SingleLeg\single_leg_02.mp4")

# frame_folder = os.path.join(frame_folder,str(folder_count))
# os.makedirs(frame_folder, exist_ok=True)

# pose_store = os.path.join(pose_store,str(folder_count))
# os.makedirs(pose_store, exist_ok=True)

pose_sequence = []
if not cap.isOpened():
    print("Failed to iterate through list")
    exit() # leave entire loop for faster
while True:

    ret,frame = cap.read()
    frame = cv2.resize(frame,(640,640)) # resize for less computation need

    if not ret or frame is None:
        print("Cant recieve frame(stream end?). Exiting.....")


    #Use yolo to iterate through frames for Pose estimation with frames can also try video
    pose_frame = model.track(frame, persist = True, stream =False)
    annotated_frame = pose_frame[0].plot()


    #can Only get confidence when still in tensor and convert to numpy()
    # Convert Confidence points per frame into numpy

    # Run YOLO pose estimation
    results = model.track(frame, persist=True, stream=False)
    annotated_frame = results[0].plot()

    # Save annotated frame
    cv2.imwrite(os.path.join(frame_folder, f"frame_{frame_count}.jpg"), annotated_frame)

    # Extract keypoints every frame
    keypoints = results[0].keypoints.xyn.cpu().numpy()  # normalized coords
    confidence = results[0].keypoints.conf.cpu().numpy()

    # If two people detected, flatten their keypoints
    if keypoints is not None and len(keypoints) == 2:
        # row = []
        # for person_kpts in keypoints:
        #   row.extend(person_kpts.flatten())  # flatten (51 points → x,y)
        #   pose_sequence.append(row)



        row = []
        for person_id in range(len(keypoints)):
            for kp_idx in range(len(keypoints[person_id])):
                x, y = keypoints[person_id][kp_idx]
                conf = confidence[person_id][kp_idx]
                row.extend([x, y, conf])  # add x, y, confidence together
            pose_sequence.append(row)
            # # When we have 30 frames, save as CSV
        if len(pose_sequence) == 30:
            df = pd.DataFrame(pose_sequence)
            df.to_csv(os.path.join(pose_store, f"video{folder_count}_pose_{frame_count}.csv"), index=False)
            pose_sequence = []  # reset for next sequence

    frame_count += 1
    if cv2.waitKey(1) == ord('q'):
        break

# folder_count += 1


#Final output is (sequence of 30 frames, 51 points , 2 people detyectectec in frame )

# [[seq 1 -> [person 1][person 2 ]]]
cap.release()
cv2.destroyAllWindows()
#Grab data to put it into numpy

    # for result in results:
    #   xyn = result.keypoints.xyn
    #   print(type(xyn))

    # cv2_imshow( pose_frame)
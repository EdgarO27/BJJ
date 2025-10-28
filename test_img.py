import numpy as np
import cv2 as cv
count = 0
cap = cv.VideoCapture('C:\Projects\AI\Image_class_bjj\Armbar by Estevan Martinez [ET3XrD-R574].f299.mp4') 
#  grabbing the video we want no sound since we are capturing body mechanics
while cap.isOpened():
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imwrite("frame%d.jpg" % count, frame)
    count = count + 1
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

# vid = cv2.VideoCapture("C:\\Users\\Admin\\PycharmProjects\\project_1\\openCV.mp4")

# count, success = 0, True
# while success:
#     success, image = vid.read() # Read frame
#     if success: 
#         cv2.imwrite(f"frame{count}.jpg", image) # Save frame
#         count += 1

# vid.release()


# yt-dlp -f "bv*[ext=.mp4]+ba[ext=.m4a]/b[ext=.mp4] / bv*+ba/b" "https://www.youtube.com/watch?v=ET3XrD-R574"


# Armbar by Estevan Martinez [ET3XrD-R574].f251.webm
        # Armbar by Estevan Martinez [ET3XrD-R574].f299.mp4
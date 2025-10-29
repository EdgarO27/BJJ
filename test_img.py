# import numpy as np
# import cv2 as cv
# count = 0
# cap = cv.VideoCapture('Armbar_by_EstevanMartinez.mp4') 
# #  grabbing the video we want no sound since we are capturing body mechanics
# while cap.isOpened():
#     ret, frame = cap.read()
    
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     cv.imwrite("./Images/frame%d.jpg" % count, frame)
#     count = count + 1
#     cv.imshow('frame', gray)
#     if cv.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv.destroyAllWindows()








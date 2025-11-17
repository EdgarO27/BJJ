import datetime
import numpy as np
import cv2 as cv

endTime = datetime.datetime.now() + datetime.timedelta(minutes=1)

count = 0
cap = cv.VideoCapture('Videos/FinishingtheSingleLegAllOptionsandCountersTalgatIlyasov.mp4') 
        #  grabbing the video we want no sound since we are capturing body mechanics
while cap.isOpened():
    ret, frame = cap.read()
    
        # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imwrite("SingleLeg/frame%d.jpg" % count, frame)
    count = count + 1
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q') or datetime.datetime.now() >= endTime:
        break

cap.release()
cv.destroyAllWindows()






# Armbar by Estevan Martinez 
# Armbar by Estevan Martinez [ET3XrD-R574].f299.mp4
# yt-dlp -f "bv*[ext=.mp4]+ba[ext=.m4a]/b[ext=.mp4] / bv*+ba/b" "https://www.youtube.com/watch?v=ET3XrD-R574"


# Finishing the single leg
# yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4] / bv*+ba/b" "https://www.youtube.com/watch?v=s3Mm2PcqwpE"
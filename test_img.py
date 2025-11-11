import numpy as np
import cv2 as cv
count = 0
image_count = 0
cap = cv.VideoCapture('Single_Leg(Talgat_Ilyasov).mp4') 
#  grabbing the video we want no sound since we are capturing body mechanics
while cap.isOpened():
    ret, frame = cap.read()
    
    # Skip some frames so we dont over do it with images 
    if count % 5 == 0:
        
    # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Turns it into grey
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        cv.imwrite("./SingleLeg_Dataset/frame%d.jpg" % image_count, frame)
        image_count = image_count + 1
    count = count + 1
    
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


#Complete hour of standup for bjj ( NOGI and GI )
# https://www.youtube.com/watch?v=utZs33FolAc








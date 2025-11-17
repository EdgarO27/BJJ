import numpy as np
import cv2 as cv
count = 0
image_count = 0
cap = cv.VideoCapture('Videos/Single_Leg(Talgat_Ilyasov.mp4') 
#  grabbing the video we want no sound since we are capturing body mechanics
while cap.isOpened():
    ret, frame = cap.read() # reading the image and bool to see if any frames left
    
    orig_h, orig_w = frame.shape[:2]
    # Skip some frames so we dont over do it with images 
    if count % 5 == 0:
        
    # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Turns it into grey
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Had to be done due to errors being one pixel number off
        h, w = frame.shape[:2]
        new_w = (w // 32) * 32
        new_h = (h // 32) * 32
        image = cv.resize(frame, (new_w, new_h))



#image with text
        annotated_image = image.copy()  # regular image
        annotated_db50_image = image.copy() # show outlines of the poly lines

        orig_image = image.copy() # normal image with default sizin
        orig_db50_image = image.copy() # image processed and text erased

        textDetectorDB50 = cv.dnn_TextDetectionModel_DB("DB_TD500_resnet50.onnx")


        # height,width = image.shape[:2]
        # inputSize = (320,320)

        conf_thresh = 0.8 # default = .8
        nms_thresh = 0.4
        bin_thresh = 0.3
        poly_thresh = 0.3 # default = .5 Lowering poly thresh helped grab all the text on the bottom left
        mean = (122.67891434, 116.66876762, 104.00698793)
        #USING DB50 
        textDetectorDB50.setBinaryThreshold(bin_thresh).setPolygonThreshold(poly_thresh)
        textDetectorDB50.setInputParams(1.0/255,(new_w,new_h), mean, True)

    # INpainted method from open cv
        inpaint_mask_db50 = np.zeros(image.shape[:2], dtype=np.uint8)

    #params that have the image processed and also cofidence values
        boxesDB50, _ = textDetectorDB50.detect(image)

        for box in boxesDB50:
            cv.fillPoly(inpaint_mask_db50, [np.array(box, np.int32)], 255)  # DB50 mask
            cv.polylines(annotated_db50_image, [np.array(box, np.int32)], isClosed=True, color=(0, 0, 255), thickness=1)  # Annotate DB50 (Red)

        inpainted_db50_image = cv.inpaint(orig_db50_image, inpaint_mask_db50, inpaintRadius=5, flags=cv.INPAINT_NS)  # DB50 only

        # fin_image = cv.resize(inpainted_db50_image,(orig_w,orig_h))
        cv.imwrite("./SingleLeg_Text_process/frame%d.jpg" % image_count, inpainted_db50_image)

        image_count = image_count + 1
    count = count + 1
    
    # cv.imshow('Single Leg', inpainted_db50_image )
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


#Complete hour of standup for bjj ( NOGI and GI )
# https://www.youtube.com/watch?v=utZs33FolAc








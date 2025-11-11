import cv2
import numpy as np

image_path = "SingleLeg_Dataset/frame0.jpg"
image1 = cv2.imread(image_path)

h, w = image1.shape[:2]
new_w = (w // 32) * 32
new_h = (h // 32) * 32
image = cv2.resize(image1, (new_w, new_h))



#image with text
annotated_image = image.copy()  # For all models
annotated_db50_image = image.copy() 

orig_image = image.copy()
orig_db50_image = image.copy()

textDetectorDB50 = cv2.dnn_TextDetectionModel_DB("DB_TD500_resnet50.onnx")


height,width = image.shape[:2]
inputSize = (height,width)

conf_thresh = 0.8
nms_thresh = 0.4
bin_thresh = 0.3
poly_thresh = 0.5
mean = (122.67891434, 116.66876762, 104.00698793)
#USING DB50 
textDetectorDB50.setBinaryThreshold(bin_thresh).setPolygonThreshold(poly_thresh)
textDetectorDB50.setInputParams(1.0/255,inputSize, mean, True)

inpaint_mask_db50 = np.zeros(image.shape[:2], dtype=np.uint8)

boxesDB50, _ = textDetectorDB50.detect(image)

for box in boxesDB50:
    cv2.fillPoly(inpaint_mask_db50, [np.array(box, np.int32)], 255)  # DB50 mask
    cv2.polylines(annotated_db50_image, [np.array(box, np.int32)], isClosed=True, color=(0, 0, 255), thickness=1)  # Annotate DB50 (Red)

inpainted_db50_image = cv2.inpaint(orig_db50_image, inpaint_mask_db50, inpaintRadius=5, flags=cv2.INPAINT_NS)  # DB50 only

cv2.imwrite("./Singleleg_text/frame0.jpg", inpainted_db50_image)
cv2.imshow('Annotated (DB50 Only)', annotated_db50_image)
cv2.imshow('Inpainted (DB50 Only)', inpainted_db50_image)


cv2.waitKey(0)
cv2.destroyAllWindows()
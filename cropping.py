
import cv2
import numpy as np 
# Load the image
img = cv2.imread("SingleLeg_Dataset/frame0.jpg")

original_height, original_width = img.shape[:2]

scale_down = 0.95
# Define new width while maintaining the aspect ratio
# new_width = 1000
# aspect_ratio = new_width / original_width
# new_height = int(original_height * aspect_ratio)
resized_image = cv2.resize(img,None, fx = scale_down, fy=scale_down, interpolation=cv2.INTER_LINEAR)

cv2.imshow("Resized",resized_image)
# Let user select ROI (drag a box)

roi = cv2.selectROI("Select ROI", resized_image, False)
 
# Extract cropped region
cropped_img = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
 
# Save and display cropped image
cv2.imwrite("Cropped.png", cropped_img)
cv2.imshow("Cropped Image", cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
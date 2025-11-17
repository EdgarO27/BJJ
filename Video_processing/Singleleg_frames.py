import cv2
import time

# Start video capture
video = cv2.VideoCapture("C:/Projects/AI/Image_class_bjj/Videos/FinishingtheSingleLegAllOptionsandCountersTalgatIlyasov.mp4")

# Get FPS using OpenCV property
fps = video.get(cv2.CAP_PROP_FPS)
ct = video.get(cv2.CAP_PROP_FRAME_COUNT)
print(f"Frames per second using video.get(cv2.CAP_PROP_FPS): {fps}")
print(f"Frames per second using video.get(cv2.CAP_PROP_FPS): {ct}")

# Calculate FPS manually
num_frames = 0; # Number of frames to capture
print(f"Capturing {num_frames} frames")

# Start time
start = time.time()
counter = 0
# Capture frames
while video.isOpened():
    ret, frame = video.read()
    
    if num_frames % 5 == 0:
        
        cv2.imwrite("C:/Projects/AI/Image_class_bjj/Singleleg/frame%d.jpg" % counter, frame)
        counter += 1
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    num_frames += 1
    
# End time
end = time.time()

# Calculate time elapsed and FPS
seconds = end - start
fps_manual = num_frames / seconds 
print(f'Number of Frames: {num_frames}')
print(f"Time taken: {seconds} seconds")
print(f"Estimated frames per second: {fps_manual}")

# Release the video capture object
video.release()
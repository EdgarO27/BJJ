import cv2

import time 
from datetime import timedelta
from datetime import datetime


# duration = 60 * 3
# seconds = timedelta(seconds= duration)

# current = timedelta(seconds= time.time())
# #duration in seconds so i just input for i defining how many minutes 


# start_time = time.time()
# end_time = time.time() * 3

# counter_time = time.perf_counter()

# print(counter_time)
# print(start_time)
# print(seconds)
# print(current)


# Function to measure elapsed time of a given task
def measure_elapsed_time(task_function, *args, **kwargs):
    # Capture start time
    start_time = time.perf_counter()

    # Execute the task
    task_function(*args, **kwargs)

    # Capture end time
    end_time = time.perf_counter()

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    # Return the elapsed time
    return elapsed_time

# Example usage
# Define a task function
def example_task():
    # Simulate a task by sleeping for a specified duration
    vid = cv2.VideoCapture("C:\\Users\\Admin\\PycharmProjects\\project_1\\openCV.mp4")
    start_time =time.perf_counter()
    elapsed_time = -1
    if( elapsed_time == 0):
    #Running till False Putting it a specfic folder 
        count, success = 0, True
        while success:
            
            success, image = vid.read() # Read frame
            if success: 
                cv2.imwrite(f"./Video2/frame{count}.jpg", image) # Save frame
                count += 1
            end_time = time.perf_counter()


    vid.release()

# Measure elapsed time for the example task
elapsed = measure_elapsed_time(example_task, 1*60) # Task duration is 2 seconds
print(f'Elapsed time: {elapsed} seconds')



# vid = cv2.VideoCapture("C:\\Users\\Admin\\PycharmProjects\\project_1\\openCV.mp4")

    
# count, success = 0, True
# while success:
#     success, image = vid.read() # Read frame
#     if success: 
#         cv2.imwrite(f"frame{count}.jpg", image) # Save frame
#         count += 1

# vid.release()


# https://www.youtube.com/watch?v=s3Mm2PcqwpE
# yt-dlp -f "bv*[ext=.mp4]+ba[ext=.m4a]/b[ext=.mp4] / bv*+ba/b" "https://www.youtube.com/watch?v=s3Mm2PcqwpE"
# 
# "https://www.youtube.com/watch?v=ET3XrD-R574"
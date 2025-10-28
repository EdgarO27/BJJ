from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model.track("Armbar_by_EstevanMartinez.mp4", show = True, save=True)  # predict on an image


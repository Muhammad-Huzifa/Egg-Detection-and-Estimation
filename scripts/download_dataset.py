from roboflow import Roboflow

rf = Roboflow(api_key="Your API KEY")
project = rf.workspace("digital-image-proecessing").project("eggs-dpy01-yqgdf")
version = project.version(1)
dataset = version.download("yolov8")
print(f"Dataset downloaded: {dataset.location}")
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

retults = model("BallTrack.mp4", show=True)
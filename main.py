from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model("test video.mp4", save=True)

print("Detection completed!")
print("The detected video has been saved.")
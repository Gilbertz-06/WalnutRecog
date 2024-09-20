
import time
import cv2
from ultralytics import YOLO

device = 0
model = YOLO("../weights/walnut_v8n.pt") #.to(device) # m=train73 l=train75, s= train76 n= train77
img_path = "../data/Walnut10.jpg"

img = cv2.imread(img_path)  # load image
model.predict(source=img, imgsz=640)

# start timer
start_time = time.time()
model.predict(source=img, imgsz=640)  # ,imgsz=640
# end timer
end_time = time.time()

# calculate time
detection_time = end_time - start_time

print(f"detection_time: {detection_time:.4f} sec")

# results = model(source=image, save=True, show_labels=False)

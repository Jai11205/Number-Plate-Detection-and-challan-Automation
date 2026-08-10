import cv2
import numpy as np

def get_plate_from_car(car_crop, model):
    results = model(car_crop, conf=0.5)  # higher confidence
    conf = 0.5

    for result in results:
            
            boxes = [b for b in result.boxes if b.conf[0] > 0.5]
            if not boxes:
                return None
            best_box = max(boxes, key=lambda b: b.conf[0])
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            # loop through all detections
            cv2.rectangle(car_crop, (x1,y1), (x2,y2), (0,255,0), 2)
            print(f"BBox: {x1},{y1},{x2},{y2} | Conf: {conf:.2f}")
            
            h, w = car_crop.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                return car_crop[y1:y2, x1:x2]
   
     
    print("No valid plate detected!")
    return None

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO



def process_video_feed_with_enforcement(video_path: str, output_path: str = '/kaggle/output/output_video.mp4'):

    best_plate_crop = None
    highest_sharpness = 0
    plate_buffer = []
    
    conn = init_database()
    
    source_points = np.array([[550, 230], [1750, 250],[0, 1000], [1780, 1000]] ,dtype=np.float32)
    # 1. Load your models
    # vehicle_model should detect cars/bikes (COCO classes 2, 3, 5, 7)
    vehicle_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    plate_model = YOLO('/kaggle/input/models/btia23015/yolov8-finetuned/pytorch/default/1/best (1).pt') # Load your custom plate model here later

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 2. Initialize Custom Components
    tracker = CentroidTracker(max_disappeared=30)

    # NOTE: Run the Matplotlib helper in Colab first to find these 4 specific pixel coordinates!
    
    # Assuming the real-world road patch you selected is roughly 10m wide and 30m long
    transformer = PerspectiveTransformer(source_points, real_world_width=10.0, real_world_length=30.0)

    # Dictionary to track vehicle history: {object_id: (real_world_coords, frame_number)}
    vehicle_history = {}
    SPEED_LIMIT_KMH = 5.0
    flagged_vehicles = set()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- A. VEHICLE DETECTION ---
        results = vehicle_model(frame)
        detections = results.xyxy[0].cpu().numpy()

        rects = []
        centroid_to_bbox = {} # NEW: Map centroids back to bounding boxes

        for det in detections:
            class_id = int(det[5])
            confidence = det[4]
            if class_id in [2, 3, 5, 7] and confidence > 0.5:
                x1, y1, x2, y2 = map(int, det[:4])
                rects.append((x1, y1, x2, y2))

                # Calculate centroid and store the mapping
                cX = int((x1 + x2) / 2.0)
                cY = int((y1 + y2) / 2.0)
                centroid_to_bbox[(cX, cY)] = (x1, y1, x2, y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # --- B. TRACKING & SPEED CALCULATION ---
        objects = tracker.update(rects)
        for (object_id, centroid) in objects.items():
            real_world_coord = transformer.transform_point(centroid)
            color = (0, 255, 0) # Default color is green

            if object_id not in vehicle_history:
                vehicle_history[object_id] = (real_world_coord, frame_count)
            else:
                prev_coord, prev_frame = vehicle_history[object_id]
                frames_elapsed = frame_count - prev_frame

                if frames_elapsed >= 15:
                    speed = calculate_speed(prev_coord, real_world_coord, fps, frames_elapsed)
                    vehicle_history[object_id] = (real_world_coord, frame_count)

                    # --- ENFORCEMENT ---
                    if speed > SPEED_LIMIT_KMH and object_id not in flagged_vehicles:
                      flagged_vehicles.add(object_id)

                      # 1. Get the BBox for the car
                      bbox = centroid_to_bbox.get(tuple(centroid))
                      if bbox:
                          x1, y1, x2, y2 = bbox
                          cropped_car = frame[y1:y2, x1:x2]
                          
                          if cropped_car is None or cropped_car.size == 0: return None
                          # 2. Use your custom YOLO model to extract the plate from the car
                          
                          plate_image = get_plate_from_car(cropped_car, plate_model)
                         
                          if plate_image is not None:

                              sharpness = get_sharpness_score(plate_image)
                              plate_buffer.append( (plate_image, sharpness) )
                              consecutive_misses = 0
                              cv2.putText(frame, "Tracking Plate...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                              
                          else:
                             # No plate detected in this frame. 
                             # If we have a buffer, the car might be leaving the screen.
                             if len(plate_buffer) > 0:
                                consecutive_misses += 1
                            
                                # If we miss the plate for 5 frames in a row, assume the car left
                                if consecutive_misses > 5:
                                    print(f"Car left screen. Evaluating {len(plate_buffer)} captured frames...")
                                    
                                    # --- THIS IS WHERE THE MAGIC HAPPENS ---
                                    # Sort the buffer by sharpness score (the second item in our tuple: x[1])
                                    # and grab the highest scoring crop
                                    best_plate_tuple = max(plate_buffer, key=lambda x: x[1])
                                    best_plate_crop = best_plate_tuple[0]
                                    best_score = best_plate_tuple[1]
                                    
                                    print(f"Selected best frame with sharpness score: {best_score:.2f}")
                                    
                                    # 3. NOW run your preprocessing and OCR on only this perfect frame
                                    cv2.imwrite("croped plate.jpg", best_plate_crop)
                                    #processed_plate = preprocess_for_ocr(best_plate_crop)
                                    plate_text = extract_text_from_plate(best_plate_crop)
                                 #   validated_plate = correct_and_validate_plate(plate_text)
                                 
                                    if plate_text:
                                      print(f'plate text {plate_text}')
                                      evidence_filename = f"evidence_{plate_text}_{frame_count}.jpg"
                                      cv2.imwrite(evidence_filename, cropped_car)
                                      register_violation_in_db(conn, plate_text, speed, evidence_filename)
                                      print(f"🚨 CHALLAN ISSUED: {plate_text} at {speed:.1f} km/h")
                                    else:
                                      print("Could not validate OCR reading.")
                                    # CLEAR the buffer so it's ready for the NEXT car
                                    plate_buffer = []
                                    consecutive_misses = 0
                               
                    cv2.putText(frame, f"ID {object_id}: {speed:.1f} km/h", (centroid[0]-20, centroid[1]-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

process_video_feed_with_enforcement("/kaggle/working/sample_traffic.mp4")
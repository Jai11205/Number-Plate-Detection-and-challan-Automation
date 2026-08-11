# Automated Traffic Enforcement, Speed Estimation & ANPR System

An end-to-end Computer Vision pipeline built with PyTorch, YOLOv8, OpenCV, and EasyOCR. The system detects vehicles in traffic video feeds, calculates their real-world speeds using Perspective Transformations (Homography) on ground contact points, crops license plates, performs character recognition (ALPR), and automatically issues digital citations (challans) into an SQLite3 database.

## Key Features🚘 Multi-Class Vehicle Tracking: 

Detects cars, buses, trucks, and motorcycles using YOLOv8 coupled with a custom CentroidTracker that handles persistent bounding box tracking.

## Perspective-Corrected Speed Estimation: 

Converts 2D camera pixels into 3D real-world coordinates using a 4-point Homography Matrix applied to vehicle tire contact points $(cX, y_2)$, completely eliminating 3D height parallax errors.

## Jitter-Free Speed Smoothing: 

Filters out high-frequency bounding box detection jitter over multi-frame windows for smooth, highly accurate km/h calculations.

## License Plate Extraction & OCR Pipeline:

Re-crops plates using a fine-tuned YOLOv8 detector.Image preprocessing via Lanczos4 upscaling, unsharp masking/sharpening, and CLAHE (Contrast Limited Adaptive Histogram Equalization).Text extraction with EasyOCR and format-aware regex post-processing.

## Automated Challan & Evidence Storage: 

Logs speeding violations, calculates fine amounts based on speed delta, saves image evidence snapshots, and updates an SQLite3 database.
## Tech Stack & DependenciesLanguage: 
### Python 3.10+Deep Learning Framework: 
    PyTorch, Ultralytics YOLOv8
### Computer Vision & Image Processing:
    OpenCV, Matplotlib, NumPyOCR 
### Engine: 
    EasyOCR
### Database: 
    SQLite3
### InstallationClone the repository:
    git clone https://github.com/Jai11205/Number-Plate-Detection-and-challan-Automation cd traffic-enforcement
    
## Install required dependencies: 
    
      pip install ultralytics easyocr opencv-python numpy matplotlib torch
    
## Calibration & Configuration:
   Before running speed estimation on a new video, you must calibrate the 4 road control points (source_points) for the Homography Matrix.
   1. Identify Road CoordinatesRun this snippet to view a frame from your source video with pixel grid lines:

    import cv2
    import matplotlib.pyplot as plt
    cap = cv2.VideoCapture("path/to/your/video.mp4")
    ret, frame = cap.read()
    cap.release()

    if ret:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.grid(True)
        plt.title("Identify 4 Road Points: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
        plt.show()
    
   2. Set Parameters in Script:

   Update the source_points matrix and real-world lane dimensions in your main execution call:
            
    # Pixel coordinates of the 4 road trapezoid corners
    source_points = np.array([
        [450, 300],  # Top-Left
        [850, 300],  # Top-Right
        [1100, 700], # Bottom-Right
        [200, 700]   # Bottom-Left
    ], dtype=np.float32)
    
    # Real-world physical dimensions of the marked road box
    REAL_WORLD_WIDTH = 3.5    # Width of lane in meters
    REAL_WORLD_LENGTH = 30.0  # Distance along road in meters
    SPEED_LIMIT = 60.0        # Speed threshold in km/h

    
# Usage Execute the main pipeline function:
    process_traffic_video:Pythonprocess_traffic_video(
    video_path="input_traffic.mp4",
    output_path="traffic_output.mp4",
    source_points=source_points,
    real_world_width=3.5,
    real_world_length=30.0,
    speed_limit=60.0
    )
# Database Schema:
 
The pipeline automatically creates and manages an  SQLite3 database (traffic_violations.db) with two relational tables:

vehicles TableColumn
| Column            | Type            | Description                               |
|-------------------|-----------------|-------------------------------------------|
| number_plate      |  TEXT(PK)       | Primary Key / Vehicle License Plate Number|
| owner_name        |  TEXT           | Registered Owner Name                     |
| violations        |  INTEGER        | Cumulative count of speeding violations   |

challan Table

|Column          | Type               |Description                                  |
|----------------|--------------------|---------------------------------------------|
|challan_id      |INTEGER (PK)        |Auto-incremented Ticket ID                   |
|number_plate    |TEXT (FK)           |References vehicles(number_plate)            |
|violation_type  |TEXT                |Type of traffic infraction (e.g., "Speeding")|
|fine_amount     |REAL                |Calculated fine in local currency            |
|evidence_path   |TEXT                |Path to stored image evidence frame          |
|date_issued     |TIMESTAMP           |Automatic system timestamp                   |


import cv2
import numpy as np

def calculate_speed(point1, point2, fps, frame_count):
    # Calculate Euclidean distance in the transformed real-world space
    distance_meters = np.linalg.norm(np.array(point2) - np.array(point1))

    # Calculate time based on video FPS
    time_seconds = frame_count / fps

    # Speed in m/s converted to km/h
    speed_kmh = (distance_meters / time_seconds) * 3.6
    return speed_kmh

def preprocess_for_ocr(plate_crop):
    # 1. Upscale
    img = cv2.resize(plate_crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LANCZOS4)

    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Median Blur: This is excellent for removing 'salt and pepper' noise 
    # that makes text look fuzzy, without losing too many edges.
    denoised = cv2.medianBlur(gray, 3)

    # 4. Sharpening: Boost the edge intensity significantly
    kernel = np.array([[-1, -1, -1], 
                       [-1,  9, -1], 
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # 5. CLAHE: Final contrast boost
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    final = clahe.apply(sharpened)
    
    return final

def get_sharpness_score(image):
    """
    Calculates the sharpness of an image. Higher score = sharper.
    """
    # Convert to grayscale if it isn't already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Calculate the variance of the Laplacian
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score
import easyocr
import re
from collections import defaultdict, deque
from ultralytics import YOLO
# Initialize the reader once. 'en' is for English, and we enable GPU for Colab.
print("Loading EasyOCR model...")
# Ensure you are using the reader properly
# Use 'craft' for detection and a more accurate recognition model
reader = easyocr.Reader(['en'], gpu=True, detect_network='craft')
print("EasyOCR loaded!")
allowed_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def extract_text_from_plate(car_crop: np.ndarray,) -> str:
    try:
        # EasyOCR handles raw BGR numpy arrays directly
        results = reader.readtext(car_crop)

        best_text = ""
        highest_prob = 0

        for (bbox, text, prob) in results:
            # Clean the text: keep only alphanumeric characters
            clean_text = re.sub(r'[^A-Za-z0-9]', '', text.strip().upper())

            # Filter out garbage reads: Plates usually have >= 4 characters
            if prob > 0.4 and len(clean_text) >= 4:
                if prob > highest_prob:
                    highest_prob = prob
                    best_text = clean_text

        return best_text
    except Exception as e:
        return ""
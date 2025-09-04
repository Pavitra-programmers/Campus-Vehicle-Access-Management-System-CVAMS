# Character_reading.py

import os
import cv2
from scipy import ndimage
import numpy as np
import easyocr
import re

# Load the pre-trained cascade classifier for number plate detection
cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

# Reuse a single EasyOCR reader instance to avoid expensive re-initialization per frame
_OCR_READER = easyocr.Reader(['en'], gpu=False)

# Simple pattern for Indian-style plates like KA01AB1234, flexible for OCR noise
_PLATE_REGEX = re.compile(r"[A-Z]{1,3}[0-9]{1,3}[A-Z]{1,3}[0-9]{1,4}")

def _preprocess_for_ocr(plate_bgr):
    # Convert to grayscale, denoise, increase contrast, then threshold
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.equalizeHist(gray)
    # Adaptive threshold helps under varying light
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 5)
    # Slight dilation to connect characters
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.dilate(th, kernel, iterations=1)
    return th

def _normalize_text(s: str) -> str:
    # Uppercase, remove spaces/hyphens/dots that OCR often inserts
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s

text = ""
def extract_num(frame):
    # Convert the frame to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Denoise the image using a median filter
    deNoised = ndimage.median_filter(gray_img, 3)
    # Apply a high pass filter using CLAHE to enhance the image contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    highPass = clahe.apply(deNoised)
    # Detect number plates using the cascade classifier
    nplate = cascade.detectMultiScale(highPass, 1.1, 4)
    
    
    # Process each detected number plate
    for (x, y, w, h) in nplate:
        wT, hT, cT = frame.shape
        a, b = (int(0.02 * wT), int(0.02 * hT))
        # Crop the detected number plate region from the original frame
        plate = frame[y + a:y + h - a, x + b:x + w - b, :]

        # Apply morphological transformations to enhance the number plate region
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)

        # Preprocess plate region for OCR
        plate_proc = _preprocess_for_ocr(plate)

        # Run OCR and select best match by confidence and regex validity
        results = _OCR_READER.readtext(plate_proc, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        best_text = None
        best_conf = 0.0
        for bbox, candidate_text, conf in results:
            normalized = _normalize_text(candidate_text)
            if not normalized:
                continue
            # Prefer texts that match plate-like pattern
            matches_pattern = bool(_PLATE_REGEX.fullmatch(normalized))
            score = conf + (0.15 if matches_pattern else 0.0) + (0.05 * min(len(normalized), 10))
            if score > best_conf:
                best_conf = score
                best_text = normalized

        if best_text:
            return best_text
    # If no number plate is detected or OCR fails, return None
    return None

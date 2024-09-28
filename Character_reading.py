# Character_reading.py

import os
import cv2
from scipy import ndimage
import numpy as np
import easyocr

# Load the pre-trained cascade classifier for number plate detection
cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
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

    # Initialize the OCR reader
    reader = easyocr.Reader(['en'], gpu=False)

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

        # Convert the cropped plate region to grayscale
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        # Perform OCR on the cropped number plate region
        text = reader.readtext(plate_gray)
        for t in text:
            _, text, _ = t  # Extract the recognized text
            # print(f"Extracted Number Plate: {text}")

            # Return the recognized number plate text
            return text

    # If no number plate is detected or OCR fails, return None
    return None

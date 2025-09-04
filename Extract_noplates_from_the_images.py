import os
import cv2
from scipy import ndimage
import numpy as np

# Load the pre-trained cascade classifier for number plate detection
cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
text = ""

# Path to save cropped images
output_folder = 'cropped_plates'
os.makedirs(output_folder, exist_ok=True)
inp = "images/"#enter the folder name in which all the images are present

def extract_num_plate(img, image_name):
    frame = cv2.imread(image_path)
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
        
        # Save the cropped number plate image to the specified folder
        filename = os.path.join(output_folder, image_name)
        cv2.imwrite(filename, plate_gray)
        print(f'Cropped number plate image saved to {filename}')
    # If no number plate is detected, return None
    return None
# Iterate through all images in the input folder
for image_name in os.listdir(inp):
    image_path = os.path.join(inp, image_name)
    if os.path.isfile(image_path):
            extract_num_plate(image_path,image_name)

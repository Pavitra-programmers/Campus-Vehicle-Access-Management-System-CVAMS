from flask import Flask, render_template, Response, jsonify
import cv2
import os
import datetime
import sqlite3
import base64
from Character_reading import *

thres = 0.45  # Threshold to detect object

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

# Reading name of classes
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Object configuration and weight structure for dnn model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# DNN model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Global variable to store the latest vehicle frame for display
latest_vehicle_frame = None
app = Flask(__name__)

def create_vehicle_table():
    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('vehicle_data.db')
    c = conn.cursor()

    # Create the vehicle_data table if it doesn't already exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            number_plate TEXT,
            timestamp TEXT
        )
    ''')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def insert_vehicle_data(number_plate):
    # Insert vehicle data into SQLite database
    conn = sqlite3.connect('vehicle_data.db')
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO vehicle_data (number_plate, timestamp) VALUES (?, ?)",
              (number_plate, timestamp))
    conn.commit()
    conn.close()


def gen_frames():
    global latest_vehicle_frame
    while True:
        success, frame = cap.read()  # Read the camera frame
        if not success:
            break
        else:
            classIds, confs, bbox = net.detect(frame, confThreshold=thres)
            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)  # Rectangle around object
                    cv2.putText(frame, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Show class name
                    cv2.putText(frame, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Show confidence

                    if classNames[classId - 1] in ["car", "motorcycle", "bus", "truck"]:
                        if confidence * 100 >= 75:
                            latest_vehicle_frame = frame.copy()  # Store the latest vehicle frame for display
                            number_plate = extract_num(frame)  # Extract the number plate directly from frame

                            if number_plate:
                                print(f"Extracted Number Plate: {number_plate}")
                                insert_vehicle_data(number_plate)  # Store in database
                            else:
                                print("No number plate detected or extraction failed.")

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest-image')
def latest_image():
    global latest_vehicle_frame
    
    if latest_vehicle_frame is not None:
        # Convert the latest frame to base64 string
        _, buffer = cv2.imencode('.jpg', latest_vehicle_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    else:
        # Return a placeholder string when no image is available
        return "No image available", 204

@app.route('/vehicle-data')
def get_vehicle_data():
    # Fetch the latest 10 number plate entries from the database
    conn = sqlite3.connect('vehicle_data.db')
    c = conn.cursor()
    c.execute("SELECT number_plate, timestamp FROM vehicle_data ORDER BY id DESC LIMIT 10")
    vehicle_data = c.fetchall()
    conn.close()

    # Return the data as a JSON response
    return jsonify(vehicle_data)

@app.route('/')
def home():
    return render_template('index.html')


if __name__ == "__main__":
    create_vehicle_table()  # Ensure the table is created
    app.run(debug=True)

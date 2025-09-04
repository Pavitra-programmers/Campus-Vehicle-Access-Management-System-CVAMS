from flask import Flask, render_template, Response, jsonify
import cv2
import os
import datetime
import sqlite3
import base64
import threading
import time
from Character_reading import *

thres = 0.45  # Threshold to detect object

cap = cv2.VideoCapture(0)
# Lower resolution to reduce CPU usage and lag
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 60)

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

def get_db_connection():
    # Create a new connection with performance-friendly pragmas
    conn = sqlite3.connect('vehicle_data.db')
    conn.execute('PRAGMA journal_mode=WAL;')
    conn.execute('PRAGMA synchronous=NORMAL;')
    conn.execute('PRAGMA temp_store=MEMORY;')
    conn.execute('PRAGMA cache_size=-20000;')  # ~20MB cache
    conn.execute('PRAGMA foreign_keys=ON;')
    return conn

def create_vehicle_table():
    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = get_db_connection()
    c = conn.cursor()

    # Create the vehicle_data table if it doesn't already exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            number_plate TEXT,
            timestamp TEXT
        )
    ''')

    # Indexes for faster reads/dedup checks
    c.execute('CREATE INDEX IF NOT EXISTS idx_vehicle_timestamp ON vehicle_data(timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_vehicle_number_plate ON vehicle_data(number_plate)')

    # On startup, ensure only today's records remain (keep DB empty each new day)
    c.execute("DELETE FROM vehicle_data WHERE date(timestamp) < date('now')")

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def insert_vehicle_data(number_plate):
    # Insert vehicle data into SQLite database
    conn = get_db_connection()
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO vehicle_data (number_plate, timestamp) VALUES (?, ?)",
              (number_plate, timestamp))
    conn.commit()
    conn.close()


def gen_frames():
    global latest_vehicle_frame
    # Throttle OCR to avoid heavy per-frame processing
    last_ocr_time_ms = 0
    # Avoid duplicate inserts: remember last plate and time
    last_plate = None
    last_plate_time_ms = 0
    min_ocr_interval_ms = 700  # run OCR at most ~1.4 times/sec
    min_reinsert_interval_ms = 30_000  # avoid re-inserting same plate within 30s
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
                            now_ms = int(time.time() * 1000)
                            # Throttle OCR work
                            if now_ms - last_ocr_time_ms >= min_ocr_interval_ms:
                                last_ocr_time_ms = now_ms
                                number_plate = extract_num(frame)  # Extract the number plate directly from frame

                                if number_plate:
                                    # Deduplicate frequent repeats
                                    if (number_plate != last_plate) or (now_ms - last_plate_time_ms >= min_reinsert_interval_ms):
                                        last_plate = number_plate
                                        last_plate_time_ms = now_ms
                                        try:
                                            insert_vehicle_data(number_plate)  # Store in database
                                            print(f"Saved plate: {number_plate}")
                                        except Exception as e:
                                            print(f"DB insert error: {e}")
                                # else: no plate detected in this throttled window

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
    conn = get_db_connection()
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

    # Background task: clear the table at midnight daily
    def daily_cleanup_worker():
        while True:
            now = datetime.datetime.now()
            # Compute seconds until midnight
            next_midnight = (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            sleep_seconds = (next_midnight - now).total_seconds()
            try:
                time.sleep(sleep_seconds)
            except Exception:
                pass
            try:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute('DELETE FROM vehicle_data')
                conn.commit()
                conn.close()
                print('Daily cleanup: vehicle_data table cleared')
            except Exception as e:
                print(f'Daily cleanup error: {e}')

    cleanup_thread = threading.Thread(target=daily_cleanup_worker, daemon=True)
    cleanup_thread.start()

    app.run(debug=True)

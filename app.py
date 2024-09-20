from flask import Flask, render_template, Response, send_from_directory
import cv2
import os
import datetime
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

app = Flask(__name__)

def gen_frames():
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
            classIds, confs, bbox = net.detect(frame, confThreshold=thres)
            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)  # rectangle
                    cv2.putText(frame, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # class name
                    cv2.putText(frame, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # confidence

                    if classNames[classId - 1] in ["car", "motorcycle", "bus", "truck"]:
                        if confidence * 100 >= 75:
                            ret, frame = cap.read()
                            if ret:
                                folder_path = 'database'
                                if not os.path.exists(folder_path):
                                    os.makedirs(folder_path)
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                saved_img = f"{classNames[classId - 1]}_{int(confidence * 100)}_{timestamp}.jpg"
                                img_name = os.path.join(folder_path, saved_img)
                                cv2.imwrite(img_name, frame)
                                print(f"Image saved successfully: {img_name}")
                                extract_num(path=img_name)
                            else:
                                print("Failed to capture the image!")
                        else:
                            print("No vehicle detected!")

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    folder_path = 'database'
    images = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
    images.sort(reverse=True)  # Sort images by latest first
    img = f"result/{images[0]}"
    extract_num(path=f"static/{img}")
    return render_template('index.html', image=img)


@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

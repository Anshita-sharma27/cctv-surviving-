import cv2
import os
import time
import numpy as np
from flask import Flask, render_template, Response
from ultralytics import YOLO
from datetime import datetime

# Flask App
app = Flask(__name__)

# Create folders
if not os.path.exists("recordings"):
    os.makedirs("recordings")

if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

# Load YOLO Model
model = YOLO("yolov8n.pt")

# Camera Setup
cap1 = cv2.VideoCapture(0)

# Check Camera
if not cap1.isOpened():
    print("Camera Not Found")
    exit()

# HD Camera Settings
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Autofocus ON
cap1.set(cv2.CAP_PROP_AUTOFOCUS, 1)

# Global Variables
status = "ROOM EMPTY"
threat = "LOW"
intruder_count = 0

recording = False
video_writer = None

last_delete_time = time.time()

# Auto Delete Old Videos
def auto_delete():

    global last_delete_time

    current = time.time()

    # Every 1 Hour
    if current - last_delete_time > 3600:

        folder = "recordings"

        for file in os.listdir(folder):

            path = os.path.join(folder, file)

            if os.path.isfile(path):

                file_time = os.path.getctime(path)

                # Delete files older than 24 hours
                if current - file_time > 86400:

                    os.remove(path)

        last_delete_time = current


# Generate Frames
def generate_frames():

    global status
    global threat
    global intruder_count
    global recording
    global video_writer

    while True:

        ret1, frame1 = cap1.read()

        if not ret1:
            print("Camera Frame Error")
            break

        # Resize
        frame1 = cv2.resize(frame1, (1280, 720))

        # Sharpen Camera
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        frame1 = cv2.filter2D(
            frame1,
            -1,
            kernel
        )

        # Better Night Vision
        frame1 = cv2.convertScaleAbs(
            frame1,
            alpha=1.0,
            beta=5
        )

        # AI Detection
        results = model(frame1)

        person_detected = False

        for result in results:

            boxes = result.boxes

            for box in boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Person Class
                if cls == 0 and conf > 0.5:

                    person_detected = True

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    # Draw Box
                    cv2.rectangle(
                        frame1,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        3
                    )

                    # Label
                    label = f"PERSON {conf:.2f}"

                    cv2.putText(
                        frame1,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

        # Current Time
        current_time = datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )

        # Time Text
        cv2.putText(
            frame1,
            current_time,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Status
        if person_detected:

            status = "INTRUDER DETECTED"
            threat = "HIGH"

            intruder_count += 1

        else:

            status = "ROOM EMPTY"
            threat = "LOW"

        # Dashboard Text
        cv2.putText(
            frame1,
            f"STATUS: {status}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame1,
            f"THREAT: {threat}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame1,
            f"COUNT: {intruder_count}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            3
        )

        # FPS
        fps = 30

        cv2.putText(
            frame1,
            f"FPS: {fps}",
            (20, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        # Camera ID
        cv2.putText(
            frame1,
            "CAMERA ID: CAM-01",
            (20, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        # Start Recording
        if person_detected and not recording:

            filename = datetime.now().strftime(
                "recordings/%Y%m%d_%H%M%S.avi"
            )

            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            video_writer = cv2.VideoWriter(
                filename,
                fourcc,
                20.0,
                (frame1.shape[1], frame1.shape[0])
            )

            recording = True

            # Save Snapshot
            image_name = datetime.now().strftime(
                "snapshots/%Y%m%d_%H%M%S.jpg"
            )

            cv2.imwrite(image_name, frame1)

        # Save Recording
        if recording:
            video_writer.write(frame1)

        # Stop Recording
        if not person_detected and recording:

            recording = False

            video_writer.release()

        # Recording Text
        if recording:

            cv2.putText(
                frame1,
                "RECORDING...",
                (1000, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

        # Auto Delete
        auto_delete()

        # Convert Frame
        ret, buffer = cv2.imencode(
            '.jpg',
            frame1
        )

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )


# Home Page
@app.route('/')
def index():

    return render_template(
        'index.html',
        status=status,
        threat=threat,
        count=intruder_count
    )


# Video Feed
@app.route('/video_feed')
def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# Run Flask Server
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
#!/usr/bin/env python3
"""
Flask + Person detection + follow-controller (pymavlink) over UART.

- Detects people using MobileNet-SSD (Caffe model).
- Streams video + detections via Flask webserver.
- Sends velocity setpoints to Pixhawk over UART (TELEM2).
- Select a target person via web UI to follow.

IMPORTANT:
 - Test with props removed first.
 - Tune DISTANCE_CALIB_K and controller gains before flight.
 - Connect Pi UART (GPIO14=TX, GPIO15=RX) to Pixhawk TELEM2 (cross TX/RX, GND common).
"""

import time
import threading
import math
import numpy as np
import cv2
from flask import Flask, render_template, Response, jsonify, request
from pymavlink import mavutil

# ---------------- Camera Config ----------------
try:
    from picamera2 import Picamera2
    HAVE_PICAM = True
except Exception:
    HAVE_PICAM = False

# ---------------- Model Config ----------------
PROTOTXT = "/home/student/PersonFollowingDrone/models/MobileNet-SSD/deploy.prototxt"
MODEL = "/home/student/PersonFollowingDrone/models/MobileNet-SSD/mobilenet_iter_73000.caffemodel"
FRAME_SIZE = (640, 480)   # width, height
CONF_THRESHOLD = 0.4
TRACK_DISTANCE_THRESHOLD = 80.0
CLASSES = ["background","aeroplane","bicycle","bird","boat",
           "bottle","bus","car","cat","chair","cow","diningtable",
           "dog","horse","motorbike","person","pottedplant",
           "sheep","sofa","train","tvmonitor"]

# ---------------- MAVLink Config ----------------
SERIAL_DEVICE = "/dev/serial0"
BAUDRATE = 57600
SEND_HZ = 10

# Follow-controller gains
MAX_FORWARD = 1.0
MAX_LAT = 0.6
MAX_YAW_DEG = 30.0
TARGET_DISTANCE_M = 3.0
KP_DIST = 0.6
KP_YAW = 18.0
KP_LAT = 0.9
DISTANCE_CALIB_K = 320.0

TYPE_MASK_VEL_ONLY = 0b0000111111000111
FRAME_BODY = 8  # MAV_FRAME_BODY_NED

# ---------------- Flask App ----------------
app = Flask(__name__)
output_frame = None
output_lock = threading.Lock()
people = []
selected_track_id = None
prev_tracks = {}
next_track_id = 1

# Load DNN
print("[INFO] Loading MobileNet-SSD model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# ---------------- Detection Loop ----------------
def detection_loop():
    global output_frame, people, prev_tracks, next_track_id, selected_track_id

    if HAVE_PICAM:
        picam2 = Picamera2()
        preview_config = picam2.create_preview_configuration(main={"size": FRAME_SIZE})
        picam2.configure(preview_config)
        picam2.start()
        print("[INFO] Using PiCamera2.")
        capture_func = lambda: picam2.capture_array()
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
        print("[INFO] Using OpenCV VideoCapture.")
        capture_func = lambda: cap.read()[1]

    while True:
        frame = capture_func()
        if frame is None:
            time.sleep(0.05)
            continue

        # Ensure BGR image
        if frame.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            frame_bgr = frame

        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        detections_list = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < CONF_THRESHOLD:
                continue
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx] if idx < len(CLASSES) else None
            if label != "person":
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w-1, endX), min(h-1, endY)
            cx = int((startX + endX) / 2)
            cy = int((startY + endY) / 2)
            detections_list.append({
                "bbox": (int(startX), int(startY), int(endX), int(endY)),
                "centroid": (cx, cy),
                "confidence": confidence
            })

        # Track assignment
        new_tracks = {}
        assigned = set()
        for track_id, prev_cent in prev_tracks.items():
            best_idx = -1
            best_dist = TRACK_DISTANCE_THRESHOLD + 1
            for j, det in enumerate(detections_list):
                if j in assigned:
                    continue
                cx, cy = det["centroid"]
                dist = math.hypot(prev_cent[0] - cx, prev_cent[1] - cy)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j
            if best_idx != -1 and best_dist <= TRACK_DISTANCE_THRESHOLD:
                detections_list[best_idx]["id"] = track_id
                new_tracks[track_id] = detections_list[best_idx]["centroid"]
                assigned.add(best_idx)

        for j, det in enumerate(detections_list):
            if j in assigned:
                continue
            det_id = next_track_id
            next_track_id += 1
            det["id"] = det_id
            new_tracks[det_id] = det["centroid"]

        prev_tracks = new_tracks

        people_now = []
        for det in detections_list:
            people_now.append({
                "id": int(det["id"]),
                "bbox": det["bbox"],
                "centroid": det["centroid"],
                "confidence": float(det["confidence"])
            })

        if selected_track_id is not None:
            if selected_track_id not in [p["id"] for p in people_now]:
                selected_track_id = None

        # Draw boxes
        for p in people_now:
            sx, sy, ex, ey = p["bbox"]
            tid = p["id"]
            conf = p["confidence"]
            color = (0, 255, 0) if tid != selected_track_id else (0, 0, 255)
            thickness = 2 if tid != selected_track_id else 3
            cv2.rectangle(frame_bgr, (sx, sy), (ex, ey), color, thickness)
            cv2.putText(frame_bgr, f"ID:{tid} {conf:.2f}", (sx, max(sy-6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        ret, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ret:
            time.sleep(0.03)
            continue

        with output_lock:
            output_frame = jpg.tobytes()
            people = people_now

        time.sleep(0.03)

# ---------------- Flask Routes ----------------
def generate_frames():
    global output_frame
    while True:
        with output_lock:
            if output_frame is None:
                continue
            frame = output_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.02)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    with output_lock:
        return jsonify({
            "width": FRAME_SIZE[0],
            "height": FRAME_SIZE[1],
            "people": people,
            "selected": selected_track_id
        })

@app.route('/select_target', methods=['POST'])
def select_target():
    global selected_track_id
    data = request.get_json(force=True)
    if not data:
        return jsonify({"success": False, "error": "no data"}), 400

    with output_lock:
        if "track_id" in data:
            tid = data["track_id"]
            if tid is None:
                selected_track_id = None
                return jsonify({"success": True, "selected": None})
            tid = int(tid)
            if any(p["id"] == tid for p in people):
                selected_track_id = tid
                return jsonify({"success": True, "selected": selected_track_id})
            else:
                return jsonify({"success": False, "error": "track_id not found"}), 404
        elif "x" in data and "y" in data:
            x = int(data["x"]); y = int(data["y"])
            chosen = None
            for p in people:
                sx, sy, ex, ey = p["bbox"]
                if sx <= x <= ex and sy <= y <= ey:
                    chosen = p
                    break
            if chosen:
                selected_track_id = int(chosen["id"])
                return jsonify({"success": True, "selected": selected_track_id})
            else:
                selected_track_id = None
                return jsonify({"success": False, "error": "no people"}), 404
        else:
            return jsonify({"success": False, "error": "bad payload"}), 400

# ---------------- Follow Thread ----------------
def estimate_distance_from_bbox_height(bbox_h_pixels):
    if bbox_h_pixels <= 0:
        return 100.0
    return DISTANCE_CALIB_K / float(bbox_h_pixels)

def clamp(v, low, high):
    return max(low, min(high, v))

def send_body_velocity(master, vx, vy, vz, yaw_rate_deg=0.0):
    ts = int(round(time.time() * 1000)) & 0xFFFFFFFF
    yaw_rate_rad = math.radians(yaw_rate_deg)
    msg = mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
        ts,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        TYPE_MASK_VEL_ONLY,
        0, 0, 0,
        float(vx), float(vy), float(vz),
        0, 0, 0,
        0.0, float(yaw_rate_rad)
    )
    print(f"[DEBUG] Sending SET_POSITION_TARGET_LOCAL_NED: "
          f"vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, yawrate={yaw_rate_deg:.2f}")
    master.mav.send(msg)

def send_statustext(master, text, severity=mavutil.mavlink.MAV_SEVERITY_INFO):
    master.mav.statustext_send(severity, text.encode("utf-8"))

def start_follow_thread():
    def thread_fn():
        try:
            master = mavutil.mavlink_connection(
                SERIAL_DEVICE,
                baud=BAUDRATE,
                source_system=200,
                source_component=190
            )

            # heartbeat sender
            def heartbeat_loop():
                while True:
                    master.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0, 0, 0
                    )
                    time.sleep(1)

            threading.Thread(target=heartbeat_loop, daemon=True).start()

            master.wait_heartbeat(timeout=30)
            print("[FOLLOW] Connected to Pixhawk (heartbeat received).")

        except Exception as e:
            print("[FOLLOW] Could not connect to Pixhawk:", e)
            return

        send_interval = 1.0 / SEND_HZ
        lost_counter = 0

        while True:
            start_time = time.time()
            with output_lock:
                sel = selected_track_id
                people_local = list(people)

            vx = vy = vz = yaw_rate = 0.0

            if sel is not None:
                target = next((p for p in people_local if p["id"] == sel), None)
                if target:
                    lost_counter = 0
                    cx, cy = target["centroid"]
                    sx, sy, ex, ey = target["bbox"]
                    bbox_h = max(1, ey - sy)
                    frame_w, frame_h = FRAME_SIZE

                    x_error = (cx - (frame_w / 2.0)) / (frame_w / 2.0)
                    dist_est = estimate_distance_from_bbox_height(bbox_h)

                    vx = clamp(KP_DIST * (dist_est - TARGET_DISTANCE_M), -MAX_FORWARD, MAX_FORWARD)
                    yaw_rate = clamp(-KP_YAW * x_error, -MAX_YAW_DEG, MAX_YAW_DEG)
                else:
                    lost_counter += 1

            # always send velocity command
            send_body_velocity(master, vx, vy, vz, yaw_rate_deg=yaw_rate)

            # debug once per second
            if int(time.time()) % 1 == 0:
                dbg = f"vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, yaw_rate={yaw_rate:.1f}"
                send_statustext(master, dbg)

            elapsed = time.time() - start_time
            time.sleep(max(0.0, send_interval - elapsed))

    t = threading.Thread(target=thread_fn, daemon=True)
    t.start()
    return t

# ---------------- Main ----------------
if __name__ == "__main__":
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    follow_thread = start_follow_thread()
    app.run(host="0.0.0.0", port=5000, threaded=True)

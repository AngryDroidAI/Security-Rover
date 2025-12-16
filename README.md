1. High‑level architecture with Qwen 0.5B

Processes on the Orange Pi 3:

    motor_control.py  
    Low-level: drive motors, turn, stop, avoid obstacles.

    vision.py  
    Camera capture, motion detection, simple “person detected” heuristic (we’ll start without heavy MobileNet to keep CPU sane).

    ai_brain.py  
    HTTP client to Ollama (qwen:0.5b), turns events into high-level decisions.

    controller.py  
    Central loop: subscribes to vision events, calls AI brain, sends commands to motor control.

    stream_server.py  
    Simple MJPEG stream for your browser.

2. Running Qwen 0.5B in Ollama

On the machine where you run Ollama (can be the Orange Pi 3 or another box):

ollama pull qwen:0.5b
ollama run qwen:0.5b


curl http://<ollama_host>:11434/api/generate -d '{
  "model": "qwen:0.5b",
  "prompt": "You are the control AI for a small security rover. Respond in JSON only."
}'


3. Motor control for L298N (Orange Pi 3 + Armbian)

Install GPIO lib (example using opi-gpio style):

sudo apt-get update
sudo apt-get install python3-pip python3-opencv
pip3 install opi-gpio requests


motor_control.py

# motor_control.py
import time
import OPi.GPIO as GPIO

# Adjust to your Orange Pi 3 pinout
MOTOR_LEFT_IN1  = 12
MOTOR_LEFT_IN2  = 13
MOTOR_RIGHT_IN1 = 6
MOTOR_RIGHT_IN2 = 7

GPIO.setmode(GPIO.BOARD)
for pin in [MOTOR_LEFT_IN1, MOTOR_LEFT_IN2, MOTOR_RIGHT_IN1, MOTOR_RIGHT_IN2]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

def _set_motor(left_forward, left_backward, right_forward, right_backward, duration=None):
    GPIO.output(MOTOR_LEFT_IN1,  GPIO.HIGH if left_forward else GPIO.LOW)
    GPIO.output(MOTOR_LEFT_IN2,  GPIO.HIGH if left_backward else GPIO.LOW)
    GPIO.output(MOTOR_RIGHT_IN1, GPIO.HIGH if right_forward else GPIO.LOW)
    GPIO.output(MOTOR_RIGHT_IN2, GPIO.HIGH if right_backward else GPIO.LOW)
    if duration:
        time.sleep(duration)
        stop()

def forward(duration=None):
    _set_motor(True, False, True, False, duration)

def backward(duration=None):
    _set_motor(False, True, False, True, duration)

def turn_left(duration=None):
    _set_motor(False, True, True, False, duration)

def turn_right(duration=None):
    _set_motor(True, False, False, True, duration)

def stop():
    _set_motor(False, False, False, False)

def cleanup():
    stop()
    GPIO.cleanup()

if __name__ == "__main__":
    try:
        forward(1)
        turn_left(0.5)
        backward(1)
        stop()
    finally:
        cleanup()


4. Vision + motion / “person-ish” detection

Start with motion detection and a very simple heuristic for “person‑like” (vertical blob). You can upgrade to a proper model later.
vision.py

# vision.py
import cv2
import time
import threading
from queue import Queue

EVENT_MOTION = "motion"
EVENT_PERSON_LIKE = "person_like"

class VisionSystem:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.prev_gray = None
        self.events = Queue()
        self.running = False

    def _process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return

        frame_delta = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray

        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = gray.shape
        motion_detected = False
        person_like = False

        for c in contours:
            if cv2.contourArea(c) < 500:
                continue
            motion_detected = True
            x, y, w, h = cv2.boundingRect(c)

            # crude heuristic: tall-ish blob in center-ish area
            aspect = h / float(w + 1e-6)
            center_x = x + w / 2
            if aspect > 1.5 and width * 0.3 < center_x < width * 0.7:
                person_like = True

        if motion_detected:
            self.events.put({"type": EVENT_MOTION, "timestamp": time.time()})
        if person_like:
            self.events.put({"type": EVENT_PERSON_LIKE, "timestamp": time.time()})

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            self._process_frame(frame)

    def get_event(self, timeout=None):
        try:
            return self.events.get(timeout=timeout)
        except:
            return None

    def read_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def stop(self):
        self.running = False
        self.cap.release()


5. Qwen 0.5B as the AI brain

ai_brain.py:

# ai_brain.py
import requests
import json

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen:0.5b"

SYSTEM_PROMPT = """
You are the control AI for a small security rover.
You receive events and must reply with a single JSON object with this shape:

{
  "action": "patrol" | "investigate" | "follow" | "hold_position" | "alert",
  "speed": "stop" | "slow" | "normal" | "fast",
  "notes": "short human-readable explanation"
}

Never include extra text. Only output JSON.
"""

def ask_brain(event):
    user_desc = f"Event: {event['type']}. Timestamp: {event.get('timestamp','unknown')}."

    payload = {
        "model": MODEL_NAME,
        "prompt": SYSTEM_PROMPT + "\n\n" + user_desc,
        "stream": False
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    text = r.json()["response"].strip()

    # Try to isolate JSON (in case model adds text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]

    try:
        return json.loads(text)
    except Exception as e:
        print("AI parsing error:", e, "raw:", text)
        # Fallback: stop
        return {
            "action": "hold_position",
            "speed": "stop",
            "notes": "fallback due to parse error"
        }

if __name__ == "__main__":
    test_event = {"type": "person_like", "timestamp": 1234567890}
    print(ask_brain(test_event))


6. Central controller: wiring vision → Qwen → motors

controller.py:

# controller.py
import time
from vision import VisionSystem, EVENT_MOTION, EVENT_PERSON_LIKE
from ai_brain import ask_brain
import motor_control as motors

def apply_action(decision):
    action = decision.get("action", "hold_position")
    speed = decision.get("speed", "stop")
    print("Decision:", decision)

    # For now, speed just picks motion duration
    if action == "patrol":
        if speed == "fast":
            motors.forward(0.5)
        elif speed == "normal":
            motors.forward(0.3)
        elif speed == "slow":
            motors.forward(0.1)
        else:
            motors.stop()
    elif action == "investigate":
        motors.forward(0.2)
    elif action == "follow":
        motors.forward(0.3)
    elif action == "alert":
        motors.stop()
        # TODO: hook into alert system
    else:  # hold_position or unknown
        motors.stop()

def main():
    vision = VisionSystem()
    vision.start()

    try:
        while True:
            event = vision.get_event(timeout=1.0)
            if not event:
                # no event, maybe slow patrol tick:
                continue

            # Only call AI on meaningful events
            if event["type"] in (EVENT_MOTION, EVENT_PERSON_LIKE):
                decision = ask_brain(event)
                apply_action(decision)

    except KeyboardInterrupt:
        pass
    finally:
        motors.cleanup()
        vision.stop()

if __name__ == "__main__":
    main()


7. Simple MJPEG video stream

stream_server.py:

# stream_server.py
from flask import Flask, Response
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture(0)

def gen():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        data = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')

@app.route('/stream')
def stream():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)

----------upgrades notes tests------------------------

Motor control for an L298N‑type driver

Vision with motion + crude “person‑like” detection

Qwen 0.5B as an Ollama brain returning structured JSON decisions

A central controller that wires all of this together

A basic MJPEG stream to watch from a browser

Add ultrasonic obstacle avoidance into motor_control or a separate safety layer

Add a real person detector (MobileNet SSD) when you’re ready to squeeze more from CPU

Add an alert system (MQTT/Telegram) wired into "action": "alert"

Move Qwen to a stronger machine on the LAN if the Orange Pi 3 struggles under load

Patrol

Guard (stationary, watching)

Follow

Investigate

Return to base

Qwen decides mode changes

Status:
- Mode: patrol
- Battery: 72%
- Last event: motion detected 4s ago
- Location: waypoint 3


{"mode": "investigate", "notes": "motion detected near waypoint 3"}

5. Add a Web Dashboard (Live Stream + Controls + Logs)

You already have MJPEG streaming.
Next step is a full dashboard:
Features

    Live video

    Manual control buttons

    Event log

    AI decisions log

    Mode selector

    Battery display

    Patrol map

6. Add Alerts (Telegram Bot or MQTT)

When Qwen decides "action": "alert", you want a real notification.
Options

    Telegram bot (fastest, easiest)

    MQTT → Home Assistant

    Email (slow)

Alert types

    Person detected

    Motion detected

    Patrol anomaly

    Low battery

    Rover stuck

7. Add “Return to Base” Behavior

Requirements

    A charging station or “home point”

    A simple beacon or AprilTag

    A “homing” behavior

What it does

When battery < 20%, Qwen can say:

{"action": "return_home", "speed": "slow"}


The rover navigates back to base using:

    Dead‑reckoning

    Ultrasonic

    Optional AprilTag on the charger

8. Add Local Recording + Event Clips

When motion or a person is detected:

    Save 5 seconds before + after

    Store clips on SD card

    Optional upload to server

9. Add a “Night Mode” (IR LEDs + Low‑light tuning)

If you want 24/7 operation:

    Add IR LED ring

    Switch camera to IR mode

    Lower exposure

    Increase motion sensitivity

10. Add a Personality Layer (Qwen‑driven)

This is optional but fun.

You can give Qwen a “character”:

    Curious

    Cautious

    Aggressive

    Playful

Upgraded architecture overview
Process layout

Proposed file tree:rover/
  core/
    motor_control.py
    ultrasonic.py
    navigation.py
    vision_mobilenet.py
    ai_brain_qwen.py
    state_manager.py
    alerts.py
  web/
    dashboard_server.py
    static/
      index.html
      app.js
      styles.css
  services/
    rover_controller.py
    video_stream.py
  systemd/
    rover.service
    rover-video.service
    rover-dashboard.service
  models/
    MobileNetSSD_deploy.prototxt
    MobileNetSSD_deploy.caffemodel

Logical layers

    Safety layer

        ultrasonic.py (obstacle detection)

        Hard overrides to stop/back away regardless of AI

    Perception layer

        vision_mobilenet.py (motion + MobileNet SSD “person” and other classes)

    State + behavior layer

        state_manager.py (mode, waypoints, battery, home position)

        navigation.py (patrol, return-to-base movement primitives)

    AI layer (Qwen)

        ai_brain_qwen.py (qwen:0.5b, JSON-only decisions + modes)

    Operator layer

        dashboard_server.py (status API + UI)

        video_stream.py (MJPEG stream)

        alerts.py (Telegram/MQTT/etc.)

    Orchestrator

        rover_controller.py (main loop: read perception + safety, call AI, apply navigation)

2. Ultrasonic safety module

core/ultrasonic.py

# core/ultrasonic.py
import time
import OPi.GPIO as GPIO

TRIG_PIN = 3
ECHO_PIN = 11

GPIO.setmode(GPIO.BOARD)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)
GPIO.output(TRIG_PIN, GPIO.LOW)

time.sleep(0.5)

def read_distance_cm(timeout=0.02):
    GPIO.output(TRIG_PIN, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, GPIO.LOW)

    start = time.time()
    while GPIO.input(ECHO_PIN) == 0:
        if time.time() - start > timeout:
            return None
    pulse_start = time.time()

    while GPIO.input(ECHO_PIN) == 1:
        if time.time() - pulse_start > timeout:
            return None
    pulse_end = time.time()

    duration = pulse_end - pulse_start
    distance = (duration * 34300) / 2.0
    return distance

def is_too_close(threshold_cm=20):
    d = read_distance_cm()
    return d is not None and d < threshold_cm


You’ll call is_too_close() in the main loop and immediately override any AI-driven motion.

3. MobileNet SSD vision module

Download MobileNetSSD_deploy.prototxt and MobileNetSSD_deploy.caffemodel into models/.

core/vision_mobilenet.py:# core/vision_mobilenet.py
import cv2
import time
from queue import Queue
import threading
import os

EVENT_MOTION = "motion"
EVENT_PERSON = "person"
EVENT_OBJECT = "object"

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

class VisionSystem:
    def __init__(self, camera_index=0, model_dir="models", conf_thresh=0.5):
        prototxt = os.path.join(model_dir, "MobileNetSSD_deploy.prototxt")
        model = os.path.join(model_dir, "MobileNetSSD_deploy.caffemodel")
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.cap = cv2.VideoCapture(camera_index)
        self.prev_gray = None
        self.events = Queue()
        self.running = False
        self.conf_thresh = conf_thresh

    def _detect_motion(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray
            return False
        frame_delta = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) >= 500:
                return True
        return False

    def _detect_objects(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        people = []
        objects = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.conf_thresh:
                continue
            idx = int(detections[0, 0, i, 1])
            if idx >= len(CLASSES):
                continue
            label = CLASSES[idx]
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            entry = {
                "label": label,
                "confidence": float(confidence),
                "box": [int(startX), int(startY), int(endX), int(endY)]
            }
            if label == "person":
                people.append(entry)
            else:
                objects.append(entry)
        return people, objects

    def _process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        motion = self._detect_motion(gray_blur)
        people, objects = self._detect_objects(frame)

        ts = time.time()
        if motion:
            self.events.put({"type": EVENT_MOTION, "timestamp": ts})
        for p in people:
            ev = {"type": EVENT_PERSON, "timestamp": ts}
            ev.update(p)
            self.events.put(ev)
        for o in objects:
            ev = {"type": EVENT_OBJECT, "timestamp": ts}
            ev.update(o)
            self.events.put(ev)

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            self._process_frame(frame)

    def get_event(self, timeout=None):
        try:
            return self.events.get(timeout=timeout)
        except:
            return None

    def read_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def stop(self):
        self.running = False
        self.cap.release()


core/navigation.py:# core/navigation.py
import time
import core.motor_control as motors
from core.state_manager import RoverState

def step_patrol(state: RoverState):
    # Very simple: move forward a bit, then treat distance heuristically
    motors.forward(0.3)
    state.update_position(0.1, 0.0)  # fake odometry
    if state.distance_to_waypoint() < 0.2:
        state.next_waypoint()

def step_guard(state: RoverState):
    motors.stop()

def step_follow(state: RoverState):
    motors.forward(0.3)
    state.update_position(0.1, 0.0)

def step_investigate(state: RoverState):
    motors.forward(0.2)
    state.update_position(0.05, 0.0)

def step_return_home(state: RoverState):
    # naive: just keep backing toward home in abstract space
    motors.backward(0.3)
    state.update_position(-0.1, 0.0)
    if state.distance_to_home() < 0.2:
        motors.stop()
        state.set_mode("guard")

def step_mode(state: RoverState):
    if state.mode == "patrol":
        step_patrol(state)
    elif state.mode == "guard":
        step_guard(state)
    elif state.mode == "follow":
        step_follow(state)
    elif state.mode == "investigate":
        step_investigate(state)
    elif state.mode == "return_home":
        step_return_home(state)
    else:
        motors.stop()


5. Qwen prompt pack and AI brain

core/ai_brain_qwen.py:# core/ai_brain_qwen.py
import requests
import json
import time

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "qwen:0.5b"

SYSTEM_PROMPT = """
You are the control AI for a small security rover.
You operate in discrete modes: patrol, guard, follow, investigate, return_home.

You will be given:
- Current mode
- Battery percentage
- Last significant event (motion, person, object)
- Distance to home
- Distance to current waypoint

You MUST respond with a single JSON object, no extra text, with this shape:

{
  "mode": "patrol" | "guard" | "follow" | "investigate" | "return_home",
  "action": "none" | "alert",
  "notes": "short explanation for logging"
}

Rules:
- If battery < 20, strongly prefer mode "return_home".
- If battery < 10, always choose "return_home" and consider "alert".
- If a person is detected while in patrol, consider "investigate" or "follow".
- If idle and no events for a long time, use "patrol".
- If already at home and return_home mode, switch to "guard".
Never output anything that is not valid JSON.
"""

def build_user_prompt(state, last_event):
    ev = last_event or {}
    return f"""
Status:
- Current mode: {state.mode}
- Battery: {state.battery_percent:.1f}
- Last event: {ev.get('type', 'none')}
- Distance to home: {state.distance_to_home():.2f}
- Distance to waypoint: {state.distance_to_waypoint():.2f}
"""

def query_ai(state, last_event):
    prompt = SYSTEM_PROMPT + "\n\n" + build_user_prompt(state, last_event)
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    text = r.json()["response"].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]
    try:
        return json.loads(text)
    except Exception as e:
        print("AI parse error:", e, "raw:", text)
        return {
            "mode": state.mode,
            "action": "none",
            "notes": "fallback due to parse error"
        }

6. Alert system (Telegram example)

core/alerts.py:# core/alerts.py
import requests
import time

TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

def send_alert(message: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("[ALERT]", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": CHAT_ID, "text": message}, timeout=5)
    except Exception as e:
        print("Alert error:", e)


7. Dashboard (HTML/JS + Flask backend)

web/dashboard_server.py:# web/dashboard_server.py
from flask import Flask, jsonify, send_from_directory
from core.state_manager import RoverState

app = Flask(__name__, static_folder="static", static_url_path="/static")
state_ref: RoverState = None  # will be set by main process

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/state")
def api_state():
    s = state_ref
    return jsonify({
        "mode": s.mode,
        "battery": s.battery_percent,
        "position": s.position,
        "home": s.home_position,
        "waypoint_index": s.current_waypoint_index
    })

@app.route("/api/set_mode/<mode>")
def api_set_mode(mode):
    s = state_ref
    s.set_mode(mode)
    return jsonify({"ok": True, "mode": s.mode})

def run_dashboard(state):
    global state_ref
    state_ref = state
    app.run(host="0.0.0.0", port=8000, threaded=True)


web/static/index.html (minimal):<!DOCTYPE html>
<html>
<head>
  <title>Rover Dashboard</title>
  <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
  <h1>Security Rover</h1>
  <div id="status"></div>
  <div>
    <button onclick="setMode('patrol')">Patrol</button>
    <button onclick="setMode('guard')">Guard</button>
    <button onclick="setMode('follow')">Follow</button>
    <button onclick="setMode('investigate')">Investigate</button>
    <button onclick="setMode('return_home')">Return Home</button>
  </div>
  <div>
    <img id="stream" src="http://ROVER_IP:5000/stream" width="480">
  </div>
  <script src="/static/app.js"></script>
</body>
</html>


web/static/app.js:async function refreshState() {
  const res = await fetch('/api/state');
  const s = await res.json();
  document.getElementById('status').innerText =
    `Mode: ${s.mode}, Battery: ${s.battery.toFixed(1)}%, Pos: ${s.position}`;
}

async function setMode(mode) {
  await fetch('/api/set_mode/' + mode);
  await refreshState();
}

setInterval(refreshState, 2000);
refreshState();


8. Main orchestrator with safety, modes, AI, navigation

services/rover_controller.py:# services/rover_controller.py
import time
from core.ultrasonic import is_too_close
from core.vision_mobilenet import VisionSystem, EVENT_PERSON, EVENT_MOTION
from core.state_manager import RoverState
from core.navigation import step_mode
from core.ai_brain_qwen import query_ai
from core import motor_control as motors
from core import alerts
from web.dashboard_server import run_dashboard
import threading

def main():
    state = RoverState()
    vision = VisionSystem()
    vision.start()

    # dashboard in background
    threading.Thread(target=run_dashboard, args=(state,), daemon=True).start()

    try:
        while True:
            # Safety first
            if is_too_close(20):
                motors.stop()
                alerts.send_alert("Obstacle too close – stopping.")
                time.sleep(0.5)
                continue

            # Pull perception events
            event = vision.get_event(timeout=0.1)
            if event:
                state.last_event = event

            # Battery simulation
            state.battery_percent = max(0.0, state.battery_percent - 0.001)

            # Periodic AI decisions
            now = time.time()
            if now - state.last_ai_update > state.ai_interval:
                decision = query_ai(state, state.last_event)
                state.set_mode(decision.get("mode", state.mode))
                if decision.get("action") == "alert":
                    alerts.send_alert(decision.get("notes", "AI requested alert"))
                state.last_ai_update = now

            # Mode-specific navigation
            step_mode(state)

    except KeyboardInterrupt:
        pass
    finally:
        motors.cleanup()
        vision.stop()

if __name__ == "__main__":
    main()


9. Video stream service (unchanged core, but split)

services/video_stream.py – same as we had, just moved into services/.
10. Systemd services for auto-boot

systemd/rover.service:[Unit]
Description=Security Rover Core Controller
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/rover
ExecStart=/usr/bin/python3 services/rover_controller.py
Restart=always

[Install]
WantedBy=multi-user.target

systemd/rover-video.service:[Unit]
Description=Security Rover Video Stream
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/rover
ExecStart=/usr/bin/python3 services/video_stream.py
Restart=always

[Install]
WantedBy=multi-user.target


systemd/rover-dashboard.service (if you keep dashboard standalone):[Unit]
Description=Security Rover Dashboard
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/rover
ExecStart=/usr/bin/python3 web/dashboard_server.py
Restart=always

[Install]
WantedBy=multi-user.target

[Unit]
Description=Security Rover Dashboard
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/rover
ExecStart=/usr/bin/python3 web/dashboard_server.py
Restart=always

[Install]
WantedBy=multi-user.target

sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rover.service rover-video.service rover-dashboard.service
sudo systemctl start rover.service rover-video.service rover-dashboard.service

11. Where to tune next

Once this is running, the most important tuning points:

    Thresholds for ultrasonic and motion

    MobileNet confidence threshold

    Qwen system prompt (you can shape the rover’s “personality” here)

    Waypoint list and home position in RoverState

    Battery model (replace the fake drain with real ADC or battery reports, if available)








# core/vision_mobilenet.py
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

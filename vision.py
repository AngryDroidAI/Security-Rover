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

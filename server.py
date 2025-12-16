# web/dashboard_server.py
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

# core/ai_brain_qwen.py
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

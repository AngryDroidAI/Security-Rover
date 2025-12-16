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

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

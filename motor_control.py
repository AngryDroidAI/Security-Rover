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


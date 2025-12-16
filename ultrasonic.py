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


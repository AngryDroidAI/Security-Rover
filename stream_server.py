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

from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
from utils.hand_tracking import get_gesture_prediction

app = Flask(__name__)

cap = cv2.VideoCapture(0)

def generate_frames():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        prediction = "Waiting..."
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
                prediction = get_gesture_prediction(hand_landmark)

        cv2.putText(frame, f'Prediction: {prediction}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

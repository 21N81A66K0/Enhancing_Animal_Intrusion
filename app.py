from flask import Flask, render_template, Response, send_file
import cv2
import numpy as np
import tensorflow as tf
import os
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('animal_detection_model.h5')

# Load labels
labels = {0: 'cow', 1: 'deer', 2: 'dog', 3: 'elephant', 4: 'lion', 5: 'tiger'}

# Set up video capture
cap = cv2.VideoCapture(0)  # Change '0' to a video file path if needed

latest_image_path = None

def send_email_alert(image_path, animal_label):
    # Email configuration
    from_address = "rudvedvaggu@gmail.com"
    to_address = "udaykiran.2688@gmail.com"
    subject = "Animal Intruder Alert!"
    body = f"An animal intruder has been detected: {animal_label}"

    msg = MIMEMultipart()
    msg['From'] = from_address
    msg['To'] = to_address
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject
    msg.attach(MIMEText(body))

    part = MIMEBase('application', "octet-stream")
    part.set_payload(open(image_path, "rb").read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f'attachment; filename="{image_path}"')
    msg.attach(part)

    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.starttls()
    smtp.login(from_address, "rtwf jyhb rwda wagm")  # Use the generated app password here
    smtp.sendmail(from_address, to_address, msg.as_string())
    smtp.quit()

def detect_animal():
    global latest_image_path
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        img_resized = cv2.resize(frame, (224, 224))
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=0)

        # Make predictions
        predictions = model.predict(img_expanded)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        label = labels[predicted_class]

        if confidence > 0.8:
            # Draw bounding box and label
            cv2.putText(frame, f'{label}: {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # Save the detected image
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            latest_image_path = f'detected_{label}_{timestamp}.jpg'
            cv2.imwrite(latest_image_path, frame)
            
            # Send email alert
            send_email_alert(latest_image_path, label)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_animal(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest_image')
def latest_image():
    if latest_image_path and os.path.exists(latest_image_path):
        return send_file(latest_image_path, mimetype='image/jpeg')
    else:
        return "No image detected yet", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

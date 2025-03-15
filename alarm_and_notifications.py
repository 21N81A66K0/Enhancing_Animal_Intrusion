import cv2
import numpy as np
import tensorflow as tf
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders
import time

# Load the trained model
model = tf.keras.models.load_model('animal_detection_model.h5')

# Load labels
labels = {0: 'cow', 1: 'deer', 2: 'dog', 3: 'elephant', 4: 'lion', 5: 'tiger'}

# Set up video capture
cap = cv2.VideoCapture(0)  # Change '0' to a video file path if needed

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

while cap.isOpened():
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

    if confidence > 0.8:  # Trigger alarm and send email if confidence is high
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_path = f'detected_{label}_{timestamp}.jpg'
        
        # Save the detected image
        cv2.imwrite(image_path, frame)
        
        # Trigger alarm sound (uncomment the following line if you have an alarm sound file)
        # playsound('alarm_sound.mp3')
        
        # Send email alert
        send_email_alert(image_path, label)
        print(f'Alert sent for {label} with confidence {confidence:.2f}')
    
    # Draw bounding box and label
    cv2.putText(frame, f'{label}: {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Animal Detection', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

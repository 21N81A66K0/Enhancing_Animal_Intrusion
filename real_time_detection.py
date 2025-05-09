import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('animal_detection_model.h5')

# Load labels
labels = {0: 'cow', 1: 'deer', 2: 'dog', 3: 'elephant', 4: 'lion', 5: 'tiger'}

# Set up video capture
cap = cv2.VideoCapture(0)  # Change '0' to a video file path if needed

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

    # Draw bounding box and label
    label = f'{labels[predicted_class]}: {confidence:.2f}'
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Animal Detection', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

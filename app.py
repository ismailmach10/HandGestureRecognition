import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model (expects GRAYSCALE images)
model = load_model("model.h5")

IMG_SIZE = 64
class_names = ["victory"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === CRITICAL FIX (MATCH TRAINING PIPELINE) ===
    # Convert BGR (webcam) → GRAYSCALE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize
    img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # Normalize
    img = img / 255.0

    # Add channel dimension (1 channel)
    img = np.expand_dims(img, axis=-1)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)
    class_id = np.argmax(prediction)
    label = class_names[class_id]

    # Display
    cv2.putText(frame, label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

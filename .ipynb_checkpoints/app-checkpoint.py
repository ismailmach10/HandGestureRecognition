import cv2
import numpy as np
import tensorflow as tf
import os

# 1. Chargement du modèle
model_path = "C:/Users/ismai/HandGestureRecognition/models/webcam_gesture_cnn.keras"
model = tf.keras.models.load_model(model_path)
classes = ["fist", "palm", "victory"]

# --- PARAMÈTRES DE STABILITÉ ---
history = [] # Pour lisser les prédictions et éviter le clignotement
CONFIDENCE_THRESHOLD = 0.90 # On augmente le seuil car le modèle est instable
MIN_WHITE_PIXELS = 3500     # Minimum de pixels blancs pour considérer qu'il y a une main

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    size = 300
    x1, y1 = (w - size) // 2, (h - size) // 2
    roi = frame[y1:y1+size, x1:x1+size]

    # --- ÉTAPE 1 : PRÉTRAITEMENT ANTI-BRUIT ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 2)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Nettoyage morphologique : supprime les petits points blancs isolés (bruit)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # --- ÉTAPE 2 : PRÉDICTION ---
    img_ready = cv2.resize(thresh, (64, 64)).reshape(1, 64, 64, 1) / 255.0
    pred = model.predict(img_ready, verbose=0)
    
    confidence = np.max(pred)
    class_id = np.argmax(pred)

    # --- ÉTAPE 3 : LOGIQUE DE DÉCISION ROBUSTE ---
    # On compte la masse de blanc pour ignorer le visage ou les petits reflets
    white_pixels = np.sum(thresh == 255)
    
    if confidence < CONFIDENCE_THRESHOLD or white_pixels < MIN_WHITE_PIXELS:
        current_res = "Inconnu"
    else:
        current_res = classes[class_id]

    # Lissage temporel : on garde les 8 dernières prédictions
    history.append(current_res)
    if len(history) > 8: history.pop(0)
    
    # La décision finale est l'état le plus fréquent dans l'historique
    final_label = max(set(history), key=history.count)

    # --- ÉTAPE 4 : AFFICHAGE ---
    color = (0, 255, 0) if final_label != "Inconnu" else (0, 0, 255)
    
    # Rectangle et texte dynamiques
    cv2.rectangle(frame, (x1, y1), (x1+size, y1+size), color, 2)
    display_text = f"{final_label}" if final_label == "Inconnu" else f"{final_label} ({confidence*100:.1f}%)"
    cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Fenêtres de contrôle
    cv2.imshow("Hand Recognition - Stable PRO", frame)
    cv2.imshow("Vue Modele (Binaire)", thresh) # Indispensable pour débugger

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
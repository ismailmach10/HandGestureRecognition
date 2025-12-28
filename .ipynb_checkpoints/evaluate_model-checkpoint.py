import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

IMG_SIZE = 64
BATCH_SIZE = 32

DATA_DIR = "data/split_dataset/test"
MODEL_PATH = "model.h5"

# Force ONLY real gesture classes
CLASS_NAMES = ["fist", "palm", "victory"]

# Load model
model = load_model(MODEL_PATH)

# Data generator
datagen = ImageDataGenerator(rescale=1./255)

test_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASS_NAMES,   # 🔥 KEY LINE
    shuffle=False
)

# Predictions
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

# === METRICS TABLE ===
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# === CONFUSION MATRIX ===
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

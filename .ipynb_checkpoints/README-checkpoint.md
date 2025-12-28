# Hand Gesture Recognition using Convolutional Neural Networks (CNN)

## 📌 Description du projet

Ce projet vise à concevoir et implémenter un système de reconnaissance de gestes de la main en temps réel à partir d’une webcam, en utilisant des techniques de vision par ordinateur et d’apprentissage profond.  
Le système repose sur un réseau de neurones convolutif (CNN) entraîné sur un dataset personnalisé capturé via webcam.

Les gestes reconnus dans ce projet sont des gestes **statiques** :
- fist
- palm
- victory

Le projet couvre l’ensemble de la chaîne :
- acquisition des données
- prétraitement
- entraînement du modèle
- évaluation expérimentale
- inférence en temps réel

---

## 🗂️ Structure du projet

HandGestureRecognition/
├── notebooks/
│ ├── 04_capture_webcam_data.ipynb
│ ├── 05_prepare_webcam_dataset.ipynb
│ ├── 06_train_webcam_cnn_clean.ipynb
│ └── 07_webcam_live_prediction.ipynb
│
├── data/
│ ├── webcam_gestures/
│ └── split_dataset/
│ ├── train/
│ ├── val/
│ └── test/
│
├── app.py
├── prepare_dataset.py
├── evaluate_model.py
├── model.h5
├── requirements.txt
└── README.md

yaml
Copy code

---

## 🧠 Description des fichiers principaux

### 📓 Notebooks
- `04_capture_webcam_data.ipynb`  
  Capture des images depuis la webcam et création du dataset brut.
- `05_prepare_webcam_dataset.ipynb`  
  Prétraitement initial des images.
- `06_train_webcam_cnn_clean.ipynb`  
  Entraînement du modèle CNN et sauvegarde du modèle (`model.h5`).
- `07_webcam_live_prediction.ipynb`  
  Tests de prédiction en temps réel dans Jupyter.

### 🐍 Scripts Python
- `app.py`  
  Script d’inférence temps réel utilisant OpenCV et le modèle entraîné.
- `prepare_dataset.py`  
  Script de séparation du dataset en ensembles train / validation / test.
- `evaluate_model.py`  
  Évaluation du modèle (classification report et matrice de confusion).

---

## ⚙️ Dépendances

Les bibliothèques nécessaires au projet sont listées dans `requirements.txt` :

tensorflow
opencv-python
numpy
matplotlib
scikit-learn
seaborn

clean
Copy code

### Installation des dépendances
```bash
pip install -r requirements.txt
🏋️‍♂️ Entraînement du modèle
Ouvrir le notebook :

stylus
Copy code
06_train_webcam_cnn_clean.ipynb
Exécuter toutes les cellules jusqu’à la sauvegarde du modèle :

python
Copy code
model.save("model.h5")
Le fichier model.h5 doit se trouver à la racine du projet.

📊 Évaluation du modèle
L’évaluation est réalisée sur l’ensemble de test (15 % des données).

Commande :

bash
Copy code
python evaluate_model.py
Résultats générés :

confusion_matrix.png

Affichage du classification report (accuracy, precision, recall, F1-score)

🎥 Inférence en temps réel (Webcam)
Pour lancer la reconnaissance de gestes en temps réel :

bash
Copy code
python app.py
Fonctionnement :

La webcam s’ouvre automatiquement

Le geste détecté est affiché sur la vidéo

Appuyer sur q pour quitter

🧪 Paramètres principaux du modèle
Taille des images : 64 × 64

Mode couleur : niveaux de gris

Optimiseur : Adam

Fonction de perte : Categorical Cross-Entropy

Batch size : 32

Nombre d’époques : 20

⚠️ Limitations
Sensibilité à l’éclairage et à l’arrière-plan

Dataset de taille limitée

Reconnaissance uniquement de gestes statiques

🚀 Perspectives
Ajout de data augmentation

Reconnaissance de gestes dynamiques (CNN + LSTM)

Utilisation de MediaPipe pour l’extraction de landmarks

Déploiement sous forme d’application desktop ou web

👤 Auteur
Projet réalisé par :
MACHHOUR ISMAIL
MALLOUK MOHAMED TAHA

Encadré par :
Mme Salma CHRIT

Année universitaire : 2025 – 2026

📜 Licence
Ce projet est réalisé dans un cadre académique.
Toute utilisation commerciale nécessite une autorisation préalable.
import os
import cv2
import numpy as np
import tensorflow as tf

# ==========================================
# LOAD MODELS
# ==========================================

moderate_model = tf.keras.models.load_model(
    "models/Final_Moderate_Detector_V2.keras"
)

normal_severe_model = tf.keras.models.load_model(
    "models/Final_Normal_Severe_Detector_Finetuned.keras"
)

print("Models Loaded Successfully")

# ==========================================
# IMAGE FOLDER
# ==========================================

IMAGE_DIR = "uploads"

# ==========================================
# TEST ALL IMAGES
# ==========================================

for fname in sorted(os.listdir(IMAGE_DIR)):

    path = os.path.join(
        IMAGE_DIR,
        fname
    )

    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224,224))

    img = img.astype(np.float32)

    img = np.expand_dims(img, axis=0)

    moderate_score = float(
        moderate_model.predict(
            img,
            verbose=0
        )[0][0]
    )

    severe_score = float(
        normal_severe_model.predict(
            img,
            verbose=0
        )[0][0]
    )

    print(
        f"{fname:20s}"
        f"  Moderate={moderate_score:.4f}"
        f"  Severe={severe_score:.4f}"
    )
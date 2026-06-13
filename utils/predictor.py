from utils.gradcam import generate_gradcam
import os
import cv2
import numpy as np
import tensorflow as tf

# ==========================================
# LOAD MODELS
# ==========================================

import os
import cv2
import numpy as np
import tensorflow as tf
print("PREDICTOR FILE LOADED")
BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)
GRADCAM_DIR = os.path.join(
    BASE_DIR,
    "gradcam_outputs"
)

os.makedirs(
    GRADCAM_DIR,
    exist_ok=True
)

normal_severe_model = tf.keras.models.load_model(
    os.path.join(
        BASE_DIR,
        "models",
        "Final_Normal_Severe_Detector_Finetuned.keras"
    )
)

moderate_model = tf.keras.models.load_model(
    os.path.join(
        BASE_DIR,
        "models",
        "Moderate_vs_Nonmoderate_DenseNet201.keras"
    )
)

print("Both Models Loaded Successfully")


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(
            f"Cannot read image: {image_path}"
        )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img,
        (224,224)
    )

    img_array = img.astype(np.float32)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_fatty_liver(image_path):

    img_array = preprocess_image(image_path)

    moderate_score = float(
        moderate_model.predict(
            img_array,
            verbose=0
        )[0][0]
    )

    severe_score = float(
        normal_severe_model.predict(
            img_array,
            verbose=0
        )[0][0]
    )

    print("Moderate Score =", moderate_score)
    print("Severe Score =", severe_score)

    filename = os.path.basename(image_path)

    gradcam_filename = (
        os.path.splitext(filename)[0]
        + "_gradcam.jpg"
    )

    gradcam_path = os.path.join(
        GRADCAM_DIR,
        gradcam_filename
    )

    MODERATE_THRESHOLD = 0.50
    SEVERE_THRESHOLD = 0.50

    if moderate_score >= MODERATE_THRESHOLD:

        final_class = "Moderate"
        confidence = moderate_score * 100

        try:

            print("Starting Moderate GradCAM...")

            generate_gradcam(
                moderate_model,
                image_path,
                gradcam_path
            )

            print("Moderate GradCAM Generated Successfully")

        except Exception as e:

            print("GradCAM ERROR:")
            print(e)

        recommendation = (
            "Reduce sugar intake, reduce fried foods, "
            "increase physical activity and consult a doctor if needed."
        )

        explanation = (
            "The AI model detected ultrasound patterns "
            "consistent with moderate hepatic steatosis."
        )

    elif severe_score >= SEVERE_THRESHOLD:

        final_class = "Severe"
        confidence = severe_score * 100

        try:

            print("Starting Severe GradCAM...")

            generate_gradcam(
                normal_severe_model,
                image_path,
                gradcam_path
            )

            print("Severe GradCAM Generated Successfully")

        except Exception as e:

            print("GradCAM ERROR:")
            print(e)

        recommendation = (
            "Consult a hepatologist, follow a strict diet plan, "
            "monitor liver health regularly and undergo further evaluation."
        )

        explanation = (
            "The AI model detected ultrasound patterns "
            "consistent with severe hepatic steatosis."
        )

    else:

        final_class = "Normal"
        confidence = (1 - severe_score) * 100

        try:

            print("Starting Normal GradCAM...")

            generate_gradcam(
                normal_severe_model,
                image_path,
                gradcam_path
            )

            print("Normal GradCAM Generated Successfully")

        except Exception as e:

            print("GradCAM ERROR:")
            print(e)

        recommendation = (
            "Maintain a healthy diet, regular exercise "
            "and periodic health checkups."
        )

        explanation = (
            "No significant ultrasound features associated "
            "with hepatic steatosis were detected."
        )

    return {

        "class": final_class,

        "confidence": round(
            confidence,
            2
        ),

        "moderate_score": round(
            moderate_score,
            4
        ),

        "severe_score": round(
            severe_score,
            4
        ),

        "recommendation": recommendation,

        "explanation": explanation,

        "gradcam_url":
        f"http://192.168.188.69:5000/gradcam/{gradcam_filename}"
    }
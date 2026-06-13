from flask import Flask
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

print("Loading Models...")

normal_severe_model = tf.keras.models.load_model(
    "models/Final_Normal_Severe_Detector_Finetuned.keras"
)

moderate_model = tf.keras.models.load_model(
    "models/Final_Moderate_Detector_Finetuned.keras"
)

print("Models Loaded Successfully!")

@app.route("/")
def home():
    return {
        "message": "Fatty Liver AI Backend Running"
    }

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
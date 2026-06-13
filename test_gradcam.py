from tensorflow.keras.models import load_model

model = load_model(
    "models/Final_Normal_Severe_Detector_Finetuned.keras"
)

base_model = model.get_layer("densenet201")

for layer in base_model.layers[-30:]:
    print(layer.name)
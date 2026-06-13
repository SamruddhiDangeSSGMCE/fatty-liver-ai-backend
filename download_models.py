import os
import gdown

os.makedirs("models", exist_ok=True)

print("Downloading Model 1...")
gdown.download(
    "https://drive.google.com/uc?id=1coNl6nzs7a6si_wjby2o0Qu7AYu456R5",
    "models/Final_Normal_Severe_Detector_Finetuned.keras",
    quiet=False
)

print("Downloading Model 2...")
gdown.download(
    "https://drive.google.com/uc?id=1Hjtxd1JfIy1nQwsaYmhfjjPlZ7I4NV84",
    "models/Moderate_vs_Nonmoderate_DenseNet201.keras",
    quiet=False
)

print("Models Downloaded Successfully")
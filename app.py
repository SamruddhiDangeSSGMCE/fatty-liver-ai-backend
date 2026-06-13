from flask import Flask, request, jsonify
from flask import send_from_directory
from utils.predictor import predict_fatty_liver
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

@app.route("/")
def home():

    return jsonify({
        "message":"Fatty Liver AI Backend Running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:

        return jsonify({
            "error": "No file uploaded"
        }), 400

    file = request.files["file"]

    if file.filename == "":

        return jsonify({
            "error": "Empty filename"
        }), 400

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    result = predict_fatty_liver(
        filepath
    )

    return jsonify(result)

@app.route("/test")
def test_page():

    return """
    <html>
    <body>

    <h2>Fatty Liver AI Test</h2>

    <form action="/predict"
          method="post"
          enctype="multipart/form-data">

        <input type="file"
               name="file">

        <br><br>

        <input type="submit"
               value="Analyze">

    </form>

    </body>
    </html>
    """
@app.route("/gradcam/<filename>")
def get_gradcam(filename):

    return send_from_directory(
        "gradcam_outputs",
        filename
    )
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
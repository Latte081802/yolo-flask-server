from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "🚀 Flask YOLOv11 Server is running!"

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Kunwari dummy result lang muna (para matest mo sa Android app)
    result = {
        "label": "Tomato Leaf Blight",
        "confidence": 0.91
    }

    return jsonify(result), 200


# ✅ Ito ang magpapatakbo kapag gusto mong i-run manually sa terminal
if __name__ == "__main__":
    # Run locally
    app.run(host="0.0.0.0", port=10000, debug=True)

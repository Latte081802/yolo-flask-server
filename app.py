
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load YOLOv11 TFLite model
interpreter = tf.lite.Interpreter(model_path= r"C:\Users\Oreol\Desktop\Plant-Detection-master\Plant-Detection-master\Plantdetection\app\flask_server\yolov11v6.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]

@app.route('/', methods=['GET'])
def home():
    return "🚀 Flask YOLOv11 Server is running!"

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image receive'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    image = image.resize((input_width, input_height))
    input_data = np.expand_dims(image, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    detections = []
    for det in output_data[0]:
        if det[4] > 0.3:
            detections.append({
                'xmin': float(det[0]),
                'ymin': float(det[1]),
                'xmax': float(det[2]),
                'ymax': float(det[3]),
                'confidence': float(det[4]),
                'class_id': int(det[5])
            })

    return jsonify(detections)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

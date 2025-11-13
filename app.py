from flask import Flask, request, jsonify
import os, io, sys, traceback
import numpy as np
import cv2

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'cnn8grps_rad1_model.h5')
model = None

def load_model_safe():
    global model
    try:
        from keras.models import load_model
        model = load_model(MODEL_PATH)
        app.logger.info("Model loaded from %s", MODEL_PATH)
    except Exception as e:
        app.logger.error("Failed to load model: %s", e)
        traceback.print_exc()

# try loading at startup (if the model file is present)
load_model_safe()

@app.route('/')
def index():
    return "Sign Language model server. Use POST /predict with form-data 'file' (image)."

@app.route('/health')
def health():
    return jsonify({'status':'ok', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error':'file not provided (use form-data field named "file")'}), 400
    f = request.files['file']
    data = f.read()
    # decode image bytes to OpenCV
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error':'could not decode image file'}), 400

    try:
        # Basic preprocessing: convert to grayscale, resize to 400x400 like original code
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (400, 400))
        # normalize to 0-1
        x = resized.astype('float32') / 255.0
        # model may expect (1,400,400,1)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=-1)
        x = np.expand_dims(x, axis=0)
    except Exception as e:
        return jsonify({'error':'preprocessing failed', 'detail': str(e)}), 500

    if model is None:
        return jsonify({'error':'model not loaded on server'}), 503

    try:
        preds = model.predict(x)
        # convert model output to list (supports probabilistic outputs)
        preds_list = preds.tolist()
        # pick argmax
        idx = int(np.argmax(preds, axis=1)[0])
        return jsonify({'prediction_index': idx, 'predictions': preds_list})
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({'error':'inference failed', 'detail': str(e), 'traceback': tb}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

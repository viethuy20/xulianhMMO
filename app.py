from flask_cors import CORS
from flask import Flask, request, render_template, json, jsonify, send_from_directory
import json
import cv2
import numpy as np
import os
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import io
import cvlib as cv

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file

import argparse




app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def main():
    return render_template('index.html')


@app.route("/api/prepare", methods=["POST"])
def prepare():
    file = request.files['file']
    
    res = preprocessing(file)
    
    cv2.imwrite("static/UPLOAD/test.png", res) # saving processed image
    
    return json.dumps({"images": res.tolist()})


@app.route('/model')
def model():
    # Sử dụng phương pháp chính xác để xác định đường dẫn
    model_json_path = os.path.join(app.root_path, 'model_js', 'model.json')
    try:
        with open(model_json_path, 'r') as json_file:
            json_data = json.load(json_file)
            return jsonify(json_data)
    except FileNotFoundError:
        return "Model JSON file not found.", 404


@app.route('/<path:path>')
def load_shards(path):
    return send_from_directory('model_js', path)


def preprocessing(file):
    
  
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, 1)
    
     # saving uploaded img
    face, confidence = cv.detect_face(img)
    print(face)
    if len(face) == 0:
        raise ValueError("Không phát hiện khuôn mặt trong hình ảnh.")  # Thêm thông báo lỗi nếu không có khuôn mặt nào được phát hiện

    for i, f in enumerate(face):
        (sX, sY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        cv2.rectangle(img, (sX,sY), (endX,endY), (0,255,0), 2)
      # Tạo mới biến `cat` cho mỗi khuôn mặt
        cat = np.copy(img[sY:endY,sX:endX])
        cat = cv2.resize(cat, (48,48))
        cat = cat.astype("float") / 255.0
        Y = sY - 10 if sY - 10 > 10 else sY + 10
        cv2.putText(img, "",(sX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
        cv2.imwrite("static/UPLOAD/1.png", img)
    return cat


if __name__ == "__main__":
    app.run()

from flask import Flask, jsonify, request
from flask_cors import CORS
from net import Net
import torch
import image

app = Flask(__name__)

# Cross Origin Resource Sharing (CORS) handling
CORS(app, resources={'/image': {"origins": "http://localhost:8080"}})

alphabet = ['N/A', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

model = Net()
model.load_state_dict(torch.load('kek.pth'))


@app.route('/image', methods=['POST'])
def image_post_request():
    img = image.convert(request.json['image'])
    pred = model(torch.tensor(img.reshape(1, 1, 28, 28)).permute(0, 1, 3, 2))
    idx = torch.argmax(pred, -1).item()
    conf = torch.max(torch.exp(pred)).item()

    return jsonify({'confidence': conf, 'letter': alphabet[idx]})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050)

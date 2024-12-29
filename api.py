from flask import Flask, jsonify, request
import torch
import pickle
import numpy as np
from model import NeuMF
app = Flask(__name__)



@app.route('/predict', methods=['GET'])
def hello_world():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

# Đặt mô hình ở chế độ đánh giá
    model.eval()
    u = request.args.get('user_id', type=int)
    i = request.args.get('item_id', type=int)
# Dữ liệu cho user ID và item ID
    user_id = torch.tensor([u])  # user ID = 10
    item_id = torch.tensor([i])  # item ID = 200

# Dự đoán
    with torch.no_grad():
        prediction = model(user_id, item_id)
    return jsonify(message=f"{prediction.item():.4f}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
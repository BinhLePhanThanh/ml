from flask import Flask, jsonify, request
import torch
import pickle
import numpy as np
from model import NeuMF
app = Flask(__name__)



@app.route('/predict', methods=['GET'])
def hello_world():
    # Tạo một instance của NeuMF
    config = {
        'latent_dim_mf': 8,
        'latent_dim_mlp': 8,
        'layers': [64, 32, 16],
        'dropout_rate_mf': 0.2,
        'dropout_rate_mlp': 0.2
    }
    num_users = 943# Số người dùng
    num_items = 1682# Số mặt hàng

    model = NeuMF(config, num_users, num_items)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    u = request.args.get('user_id', type=int)
    i = request.args.get('item_id', type=int)

    user_id = torch.tensor([u])
    item_id = torch.tensor([i])

    with torch.no_grad():
        prediction = model(user_id, item_id)

    return jsonify(message=f"{prediction.item():.4f}")
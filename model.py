from flask import Flask, jsonify, request
import torch
import pickle
import numpy as np
class NeuMF(torch.nn.Module):
    def __init__(self, config, num_users, num_items):
        super(NeuMF, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.config = config
        
        # Phần ma trận phân tách (MF)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=config['latent_dim_mf'])
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=config['latent_dim_mf'])
        
        # Phần MLP
        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=config['latent_dim_mlp'])
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=config['latent_dim_mlp'])
        
        self.fc_layers = torch.nn.ModuleList()
        input_size = config['latent_dim_mlp'] * 2  # Kích thước đầu vào cho lớp đầu tiên (user + item embeddings)
        
        for out_size in config['layers']:
            self.fc_layers.append(torch.nn.Linear(input_size, out_size))
            input_size = out_size  # Cập nhật kích thước đầu vào cho lớp tiếp theo

        self.logits = torch.nn.Linear(in_features=config['layers'][-1] + config['latent_dim_mf'], out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        # Phần MF
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)
        mf_vector = torch.nn.Dropout(self.config['dropout_rate_mf'])(mf_vector)

        # Phần MLP        
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)

        for layer in self.fc_layers:
            mlp_vector = layer(mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)
        
        mlp_vector = torch.nn.Dropout(self.config['dropout_rate_mlp'])(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.logits(vector)
        output = self.sigmoid(logits)
        return output


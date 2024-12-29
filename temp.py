import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import pickle
# Định nghĩa lớp NeuMF
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

# Đọc dữ liệu từ file
file_path = 'E:\\temp\\ml-100k\\u.data'
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(file_path, sep='\t', header=None, names=column_names)

# Chọn các cột cần thiết
df = df[['user_id', 'item_id', 'rating']]
df['rating'] = (df['rating'] > 3).astype(int)  # Chuyển đổi rating thành nhãn nhị phân
df['user_id'] -= 1
df['item_id'] -= 1
# Tạo Dataset
class RecommendationDataset(Dataset):
    def __init__(self, dataframe):
        self.users = torch.tensor(dataframe['user_id'].values)
        self.items = torch.tensor(dataframe['item_id'].values)
        self.labels = torch.tensor(dataframe['rating'].values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.users[index], self.items[index], self.labels[index]

# Khởi tạo dataset và dataloader
dataset = RecommendationDataset(df)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Cấu hình mô hình
config = {
    'layers': [128, 64, 32],
    'latent_dim_mf': 16,
    'latent_dim_mlp': 16,
    'dropout_rate_mf': 0.2,
    'dropout_rate_mlp': 0.2
}

# Số lượng người dùng và mục tiêu
num_users = df['user_id'].nunique()
num_items = df['item_id'].nunique()
print(num_users )
print(num_items )
# Khởi tạo mô hình
model = NeuMF(config, num_users, num_items)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for user_indices, item_indices, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(user_indices, item_indices)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')

torch.save(model.state_dict(), 'model.pth')
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score

# 固定 seed
seed = 60
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 讀取資料
df = pd.read_csv("./data/processed/risk_early_warning.csv")
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
features = [col for col in df.columns if col not in ['Dates', 'Peak', 'Drawdown', 'Crash_15pct']]
target = 'Crash_15pct'

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
df_scaled = pd.DataFrame(X_scaled, columns=features)
df_scaled[target] = df[target].values

# 時間視窗
def create_sequences(data, target_col, window=60):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data.iloc[i:i+window].drop(columns=target_col).values)
        y.append(data.iloc[i+window][target_col])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(df_scaled, target, window=60)
split_idx = int(len(X_seq) * 0.85)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
g = torch.Generator()
g.manual_seed(seed)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, generator=g)

# Transformer 模型
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model=32, nhead=8, num_layers=1 , dropout=0.2):
        super(TimeSeriesTransformer, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x[-1, :, :]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

model = TimeSeriesTransformer(input_size=X_train.shape[2]).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Class weights
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

# 訓練 loop
for epoch in range(30):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# 測試
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).cpu().numpy()
    y_true = y_test_tensor.cpu().numpy()

auc = roc_auc_score(y_true, y_pred)
print(f"Test AUC: {auc:.4f}")

# 儲存
np.save('y_true.npy', y_true)
np.save('y_pred_transformer.npy', y_pred)

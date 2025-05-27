import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 固定 seed
seed = 71
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
# 設定 TensorFlow deterministic 行為
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 可選：禁用 GPU 以強化重現性

# 讀取資料
df = pd.read_csv("./data/processed/risk_early_warning.csv")
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')

features = [col for col in df.columns if col not in ['Dates', 'Peak', 'Drawdown', 'Crash_15pct']]
target = 'Crash_15pct'

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
X_df = pd.DataFrame(X_scaled, columns=features)
X_df[target] = df[target].values

# 建立 LSTM 時間視窗
def create_sequences(data, target_col, window=60):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data.iloc[i:i+window].drop(columns=target_col).values)
        y.append(data.iloc[i+window][target_col])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(X_df, target , window=60)

# 訓練測試集切分
split_idx = int(len(X_seq) * 0.85)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

# 確保測試集有至少2個崩跌樣本
while y_test.tolist().count(1.0) < 2:
    split_idx -= 1
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

from collections import Counter
print(f"Train: {Counter(y_train)}, Test: {Counter(y_test)}")

# LSTM baseline
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# Class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2,
          class_weight={0: class_weights[0], 1: class_weights[1]},
          shuffle=True)

# 測試集預測
y_pred = model.predict(X_test).flatten()
# 測試集真實值
y_true = y_test

# 測試集評估
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_pred)
print(f"Test AUC: {auc:.4f}")

# 儲存
np.save('y_true.npy', y_true)
np.save('y_pred_lstm.npy', y_pred)

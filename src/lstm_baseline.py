import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.utils import class_weight

# 讀取資料
df = pd.read_csv("./data/processed/risk_early_warning.csv")

# 加入週期性特徵
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
#df['month_sin'] = np.sin(2 * np.pi * df['Dates'].dt.month / 12)
#df['month_cos'] = np.cos(2 * np.pi * df['Dates'].dt.month / 12)
#df['dow_sin'] = np.sin(2 * np.pi * df['Dates'].dt.dayofweek / 7)
#df['dow_cos'] = np.cos(2 * np.pi * df['Dates'].dt.dayofweek / 7)

# 特徵欄位
features = [col for col in df.columns if col not in ['Dates', 'Peak', 'Drawdown', 'Crash_15pct']]

#features = ['rfsi_credit', 'dxy', 'rfsi', 'dxy_gold_ratio', 
#            'cpi_core_diff', 'gold', 'spread_10y_2y', 'ted_spread', 'cpi_yoy']


#df['Crash_15pct_lead'] = df['Crash_15pct'].shift(-30)
#df['Crash_15pct_lead'] = df['Crash_15pct_lead'].fillna(0)

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

# 標籤分布
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
          class_weight={0: class_weights[0], 1: class_weights[1]})

seed =42

# 測試集評估
loss, auc = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test AUC: {auc:.4f}")

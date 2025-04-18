import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os

# Load a manageable subset (adjust nrows if needed)
print("🔁 Loading Avazu data...")
df = pd.read_csv("train.csv", nrows=1_000_000)  # try more rows if your system can handle

# Feature Engineering
print("🔧 Preprocessing...")
df['hour_of_day'] = df['hour'].astype(str).str[-2:].astype(int)
df['day_of_week'] = df['hour'].astype(str).str[:2].astype(int) // 4 % 7

# CTR encodings
device_model_ctr = df.groupby('device_model')['click'].mean().to_dict()
device_type_ctr = df.groupby('device_type')['click'].mean().to_dict()

df['device_model_ctr'] = df['device_model'].map(device_model_ctr)
df['device_type_ctr'] = df['device_type'].map(device_type_ctr)

# Hashed features
df['device_model_hash'] = df['device_model'].apply(lambda x: hash(x) % 1_000_000)
df['banner_x_device'] = df['banner_pos'].astype(str) + "_" + df['device_type'].astype(str)
df['banner_x_device_hash'] = df['banner_x_device'].apply(lambda x: hash(x) % 100_000)

# Drop rows with missing values
df = df.dropna()

# Select final feature set
features = [
    'hour_of_day', 'day_of_week', 'banner_pos', 'device_type',
    'device_model_hash', 'device_model_ctr', 'device_type_ctr', 'banner_x_device_hash'
]
X = df[features]
y = df['click']

# Train/Val Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
print("Training LightGBM...")
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'learning_rate': 0.1
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

# Save model
os.makedirs("model", exist_ok=True)
model.save_model("model/ctr_model.txt")
print("Model saved to model/ctr_model.txt")

# Save CTR encodings for prediction
os.makedirs("utils", exist_ok=True)
import json
with open("utils/ctr_encoding_map.json", "w") as f:
    json.dump({
        "device_model": device_model_ctr,
        "device_type": device_type_ctr
    }, f)
print("CTR encodings saved to utils/ctr_encoding_map.json")

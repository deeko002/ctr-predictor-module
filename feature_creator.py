import pandas as pd
import hashlib
import json
import os

# Load CTR encoding map (fallback if missing)
ENCODING_PATH = os.path.join("utils", "ctr_encoding_map.json")
if os.path.exists(ENCODING_PATH):
    with open(ENCODING_PATH, 'r') as f:
        CTR_ENCODING = json.load(f)
else:
    CTR_ENCODING = {
        "device_model": {"model_abc": 0.02, "model_xyz": 0.08},
        "device_type": {"1": 0.13, "0": 0.07}
    }

# Default fallback values
DEFAULTS = {
    'hour': 1300,
    'banner_pos': 0,
    'device_type': 1,
    'device_model': 'model_abc',
    'site_id': 'site_001',
    'app_id': 'app_001'
}

def hash_category(value, mod=1e6):
    return int(hashlib.md5(str(value).encode()).hexdigest(), 16) % int(mod)

def create_features(raw):
    if isinstance(raw, dict):
        df = pd.DataFrame([raw])
    elif isinstance(raw, pd.DataFrame):
        df = raw.copy()
    else:
        raise ValueError("Input must be dict or DataFrame")

    # Ensure all expected columns are present
    for col, default_val in DEFAULTS.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = df[col].apply(lambda x: default_val if pd.isna(x) or x == "" or x is None else x)

    # Time features
    df['hour_of_day'] = df['hour'].astype(str).str[-2:].astype(int)
    df['day_of_week'] = df['hour'].astype(str).str[:2].astype(int) // 4 % 7

    # CTR encodings
    model_ctr_map = CTR_ENCODING.get("device_model", {})
    device_ctr_map = CTR_ENCODING.get("device_type", {})

    model_ctr_avg = sum(model_ctr_map.values()) / max(1, len(model_ctr_map))
    device_ctr_avg = sum(device_ctr_map.values()) / max(1, len(device_ctr_map))

    df['device_model_ctr'] = df['device_model'].map(model_ctr_map).fillna(model_ctr_avg)
    df['device_type_ctr'] = df['device_type'].astype(str).map(device_ctr_map).fillna(device_ctr_avg)

    # Hashed features
    df['device_model_hash'] = df['device_model'].apply(lambda x: hash_category(x))
    df['banner_x_device'] = df['banner_pos'].astype(str) + "_" + df['device_type'].astype(str)
    df['banner_x_device_hash'] = df['banner_x_device'].apply(lambda x: hash_category(x, mod=100_000))

    # Count features
    df['site_id_count'] = df['site_id'].map(df['site_id'].value_counts()) if 'site_id' in df.columns else 0
    df['app_id_count'] = df['app_id'].map(df['app_id'].value_counts()) if 'app_id' in df.columns else 0

    # Hour CTR (static fallback to global mean CTR if needed)
    global_avg_ctr = 0.174
    df['hour_avg_click'] = df['hour_of_day'].map(df.groupby('hour_of_day')['device_type'].count()).fillna(global_avg_ctr)

    expected_features = [
        'hour_of_day',
        'day_of_week',
        'banner_pos',
        'device_type',
        'device_model_hash',
        'device_model_ctr',
        'device_type_ctr',
        'banner_x_device_hash',
        'site_id_count',
        'app_id_count',
        'hour_avg_click'
    ]
    return df[expected_features]

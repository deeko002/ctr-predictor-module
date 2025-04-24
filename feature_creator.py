import pandas as pd
import hashlib
import json
import os

# Load CTR encoding map
ENCODING_PATH = os.path.join("utils", "ctr_encoding_map.json")
if os.path.exists(ENCODING_PATH):
    with open(ENCODING_PATH, 'r') as f:
        CTR_ENCODING = json.load(f)
else:
    CTR_ENCODING = {
        "device_model": {"model_abc": 0.02},
        "device_type": {"1": 0.13},
        "site_id_count": {"site_001": 50},
        "app_id_count": {"app_001": 100},
        "hour_avg_click": {str(i): 0.17 for i in range(24)}  # default if hour maps missing
    }

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

    for col, default_val in DEFAULTS.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = df[col].apply(lambda x: default_val if pd.isna(x) or x == "" else x)

    # Time features
    df['hour_of_day'] = df['hour'].astype(str).str[-2:].astype(int)
    df['day_of_week'] = df['hour'].astype(str).str[:2].astype(int) // 4 % 7

    # CTR encodings
    model_ctr = CTR_ENCODING.get("device_model", {})
    type_ctr = CTR_ENCODING.get("device_type", {})
    df['device_model_ctr'] = df['device_model'].map(model_ctr).fillna(0.02)
    df['device_type_ctr'] = df['device_type'].astype(str).map(type_ctr).fillna(0.13)

    # Hashes
    df['device_model_hash'] = df['device_model'].apply(lambda x: hash_category(x))
    df['banner_x_device'] = df['banner_pos'].astype(str) + "_" + df['device_type'].astype(str)
    df['banner_x_device_hash'] = df['banner_x_device'].apply(lambda x: hash_category(x, mod=100_000))

    # ✅ Load global count encodings from training
    site_count_map = CTR_ENCODING.get("site_id_count", {})
    app_count_map = CTR_ENCODING.get("app_id_count", {})
    hour_avg_click = CTR_ENCODING.get("hour_avg_click", {})

    df['site_id_count'] = df['site_id'].map(site_count_map).fillna(50)
    df['app_id_count'] = df['app_id'].map(app_count_map).fillna(100)
    df['hour_avg_click'] = df['hour_of_day'].astype(str).map(hour_avg_click).fillna(0.174)

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

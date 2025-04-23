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
    'device_model': 'model_abc'
}

def hash_category(value, mod=1e6):
    return int(hashlib.md5(str(value).encode()).hexdigest(), 16) % int(mod)

def create_features(raw):
    """
    Accepts a dict (single row) or a DataFrame (multiple rows),
    returns model-ready numeric DataFrame with default fallback
    """
    if isinstance(raw, dict):
        df = pd.DataFrame([raw])
    elif isinstance(raw, pd.DataFrame):
        df = raw.copy()
    else:
        raise ValueError("Input must be dict or DataFrame")

    # Ensure required columns exist and fill with defaults if missing or empty
    for col, default_val in DEFAULTS.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = df[col].apply(lambda x: default_val if pd.isna(x) or x == "" or x is None else x)

    # Time-based features
    df['hour_of_day'] = df['hour'].astype(str).str[-2:].astype(int)
    df['day_of_week'] = df['hour'].astype(str).str[:2].astype(int) // 4 % 7

    # CTR encodings with fallback to mean
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

    # Return numeric model-ready features only
    expected_features = [
        'hour_of_day',
        'day_of_week',
        'banner_pos',
        'device_type',
        'device_model_hash',
        'device_model_ctr',
        'device_type_ctr',
        'banner_x_device_hash'
    ]

    return df[expected_features]

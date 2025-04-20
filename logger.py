import pandas as pd
import os
from datetime import datetime

# Save log file relative to this script's location
LOG_PATH = os.path.join(os.path.dirname(__file__), "prediction_log.csv")

def load_log():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    else:
        return pd.DataFrame(columns=[
            "timestamp", "hour", "banner_pos", "device_type", "device_model", "predicted_ctr"
        ])

def log_prediction(input_dict, predicted_ctr):
    df = load_log()

    new_row = input_dict.copy()
    new_row["predicted_ctr"] = predicted_ctr
    new_row["timestamp"] = datetime.now().isoformat()

    new_row_ordered = {
        "timestamp": new_row.get("timestamp"),
        "hour": new_row.get("hour"),
        "banner_pos": new_row.get("banner_pos"),
        "device_type": new_row.get("device_type"),
        "device_model": new_row.get("device_model"),
        "predicted_ctr": new_row.get("predicted_ctr")
    }

    df = pd.concat([df, pd.DataFrame([new_row_ordered])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    return df

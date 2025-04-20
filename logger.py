# logger.py

import pandas as pd
import os

LOG_PATH = "prediction_log.csv"

def load_log():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    else:
        return pd.DataFrame(columns=[
            "hour", "banner_pos", "device_type", "device_model", "predicted_ctr"
        ])

def log_prediction(input_dict, predicted_ctr):
    df = load_log()
    new_row = input_dict.copy()
    new_row["predicted_ctr"] = predicted_ctr
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    return df

# predict_ctr.py

import lightgbm as lgb
from feature_creator import create_features
from logger import log_prediction


# Load trained model
import os

model_path = os.path.join(os.path.dirname(__file__), "model", "ctr_model.txt")
model = lgb.Booster(model_file=model_path)

def predict_ctr(input_dict):
    # print("⚙️ Running predict_ctr...")
    features = create_features(input_dict)
    prediction = round(model.predict(features)[0], 6)

    # ✅ Log the prediction
    # print("📝 Logging prediction now...")
    log_prediction(input_dict, prediction)

    return prediction

# Example usage
if __name__ == "__main__":
    sample_input = {
        'hour': 1430,
        'device_type': 1,
        'banner_pos': 1,
        'device_model': 'model_xyz'
    }
    print("Predicted CTR:", predict_ctr(sample_input))

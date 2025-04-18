# predict_ctr.py

import lightgbm as lgb
from feature_creator import create_features

# Load trained model
model = lgb.Booster(model_file='model/ctr_model.txt')

def predict_ctr(user_input: dict) -> float:
    features = create_features(user_input)
    return round(model.predict(features)[0], 6)

# Example usage
if __name__ == "__main__":
    sample_input = {
        'hour': 1430,
        'device_type': 1,
        'banner_pos': 1,
        'device_model': 'model_xyz'
    }
    print("Predicted CTR:", predict_ctr(sample_input))

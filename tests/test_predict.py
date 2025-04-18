# tests/test_predict.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from predict_ctr import predict_ctr

def test_prediction():
    sample_input = {
        'hour': 1400,
        'device_type': 1,
        'banner_pos': 1,
        'device_model': 'model_abc'
    }
    pred = predict_ctr(sample_input)
    print("Predicted CTR:", pred)
    assert 0 <= pred <= 1, "Prediction should be a valid probability"

# Run test
if __name__ == "__main__":
    test_prediction()

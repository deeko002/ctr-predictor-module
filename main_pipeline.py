import pandas as pd
import duckdb
from feature_creator import create_features
from predict_ctr import model

# Load Avazu data once globally
avazu_df = pd.read_csv("train.csv", nrows=1_000_000)

def run_sql_to_ctr_predictions(sql_query):
    """
    Takes a SQL query string, runs it on avazu_df,
    converts result to model features, returns CTR predictions
    """
    try:
        # Step 1: Run SQL on avazu_df
        raw_df = duckdb.query_df(avazu_df, "avazu_df", sql_query).to_df()

        if raw_df.empty:
            return "No results from query."

        # Step 2: Convert to model-ready features
        features_df = create_features(raw_df)

        # Step 3: Run prediction
        ctr_preds = model.predict(features_df)

        # Step 4: Add predictions to original data
        raw_df["predicted_ctr"] = ctr_preds

        return raw_df[["hour", "banner_pos", "device_type", "device_model", "predicted_ctr"]]

    except Exception as e:
        return f"Error: {str(e)}"

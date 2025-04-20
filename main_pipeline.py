import pandas as pd
import duckdb
from feature_creator import create_features
from predict_ctr import predict_ctr  # ✅ Use your wrapped function

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

        # Step 2: Add predictions row-by-row (with logging)
        def row_to_ctr(row):
            input_dict = {
                "hour": row["hour"],
                "banner_pos": row["banner_pos"],
                "device_type": row["device_type"],
                "device_model": row["device_model"]
            }
            return predict_ctr(input_dict)

        raw_df["predicted_ctr"] = raw_df.apply(row_to_ctr, axis=1)

        return raw_df[["hour", "banner_pos", "device_type", "device_model", "predicted_ctr"]]

    except Exception as e:
        return f"Error: {str(e)}"

import pandas as pd
import duckdb
from feature_creator import create_features
from predict_ctr import model

# Required columns and fallback values
REQUIRED_COLUMNS = ["hour", "banner_pos", "device_type", "device_model"]
DEFAULT_ROW = {
    "hour": 1300,
    "banner_pos": 0,
    "device_type": 1,
    "device_model": "model_abc"
}

def run_sql_to_ctr_predictions(sql_query, df):
    """
    Takes a SQL query string and a Spark or Pandas DataFrame,
    runs the query using DuckDB, generates model-ready features,
    and returns predicted CTR values.
    """
    try:
        # If Spark DF, convert to Pandas
        if hasattr(df, "toPandas"):
            df = df.toPandas()

        # Run SQL using DuckDB
        raw_df = duckdb.query_df(df, "avazu_df", sql_query).to_df()

        if raw_df.empty:
            return "No results from query."

        # Fill in missing model-required columns
        for col in REQUIRED_COLUMNS:
            if col not in raw_df.columns:
                raw_df[col] = DEFAULT_ROW[col]
            else:
                raw_df[col] = raw_df[col].fillna(DEFAULT_ROW[col])

        # Create model features
        features_df = create_features(raw_df)

        # Predict CTR
        ctr_preds = model.predict(features_df)

        # Add prediction column
        raw_df["predicted_ctr"] = ctr_preds

        return raw_df[["hour", "banner_pos", "device_type", "device_model", "predicted_ctr"]]

    except Exception as e:
        return f"Error: {str(e)}"

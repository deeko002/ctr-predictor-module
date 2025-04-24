# main_pipeline.py

import pandas as pd
import lightgbm as lgb
from feature_creator import create_features

# What your model always needs
REQUIRED_COLUMNS = ["hour", "banner_pos", "device_type", "device_model"]
# If the user only returns one of these, we'll simulate the rest
SIMULATABLE_FEATURES = {
    "hour", "banner_pos", "device_type", "device_model",
    "day_of_week", "hour_of_day", "app_category"
}

# Fallback values when SQL didn’t return them
DEFAULT_ROW = {
    "hour":         1300,
    "banner_pos":     0,
    "device_type":    1,
    "device_model": "model_abc"
}

# Lazy-loaded global
_booster = None
def _load_model():
    global _booster
    if _booster is None:
        _booster = lgb.Booster(model_file="/databricks/driver/ctr-predictor-module/model/ctr_model.txt")


def run_sql_to_ctr_predictions(sql_query: str, spark_df) -> pd.DataFrame:
    """
    1. Runs SQL against `spark_df` (temp view name ctr_df).
    2. Fills in any missing required cols.
    3. If all required cols came back, direct predict per row.
       Else if exactly one simulatable col came back, simulate the rest 
       via the global “mode” + that column.
    4. Otherwise, predict a single global CTR and append to every row.

    Always returns only the original SQL columns + `predicted_ctr`.
    """
    # 1) Register and run your SQL
    spark_df.createOrReplaceTempView("ctr_df")
    clean_sql = sql_query.replace("__THIS__", "ctr_df")
    pdf = spark_df.sparkSession.sql(clean_sql).toPandas()

    # remember what the user asked for
    request_cols = list(pdf.columns)

    # 1a) If no rows, use a single default row
    if pdf.empty:
        pdf = pd.DataFrame([DEFAULT_ROW.copy()])
        # in this case the user-requested columns are all of our defaults
        request_cols = list(DEFAULT_ROW.keys())

    # 1b) Drop any old predicted_ctr (so we never double-append)
    pdf = pdf.drop(
        columns=[c for c in pdf.columns if c.lower() == "predicted_ctr"],
        errors="ignore"
    )
    # and keep our request_cols in sync
    request_cols = [c for c in request_cols if c.lower() != "predicted_ctr"]

    # 2) Ensure we have at least the defaults for your required cols
    for col, default in DEFAULT_ROW.items():
        if col not in pdf.columns:
            pdf[col] = default
        else:
            pdf[col] = pdf[col].fillna(default)

    # 3) Decide how to predict
    _load_model()

    # 3a) If the SQL returned *all* the required columns -> direct predict
    if all(col in pdf.columns for col in REQUIRED_COLUMNS):
        feats = create_features(pdf)
        pdf["predicted_ctr"] = _booster.predict(feats)
        return pdf[request_cols + ["predicted_ctr"]]

    # 3b) If exactly one simulatable feature came back, do a “what-if” sweep
    sim_cols = [c for c in pdf.columns if c in SIMULATABLE_FEATURES]
    if len(sim_cols) == 1:
        sim_col = sim_cols[0]

        # Grab the “typical” row from the full dataset
        mode_row = spark_df.toPandas().mode(numeric_only=False).iloc[0].to_dict()
        out_rows = []
        for _, row in pdf.iterrows():
            sim_input = mode_row.copy()
            sim_input[sim_col] = row[sim_col]
            sim_df = pd.DataFrame([sim_input])
            feats = create_features(sim_df)
            pred = _booster.predict(feats)[0]

            # preserve the original agg-row and append our prediction
            row_dict = row.to_dict()
            row_dict["predicted_ctr"] = pred
            out_rows.append(row_dict)

        result_df = pd.DataFrame(out_rows)
        # sort so highest-CTR comes first
        result_df = result_df.sort_values(by="predicted_ctr", ascending=False)
        return result_df[request_cols + ["predicted_ctr"]]

    # 3c) Otherwise, fall back to a single global prediction
    mode_row = spark_df.toPandas().mode(numeric_only=False).iloc[0].to_dict()
    feats = create_features(pd.DataFrame([mode_row]))
    pred = _booster.predict(feats)[0]
    pdf["predicted_ctr"] = pred
    return pdf[request_cols + ["predicted_ctr"]]

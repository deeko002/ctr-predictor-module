# main_pipeline.py

import os
import pandas as pd
import lightgbm as lgb
from feature_creator import create_features
import openai

# ————————————————————————————————
# 1) your existing constants & loading logic
# ————————————————————————————————
REQUIRED_COLUMNS = ["hour", "banner_pos", "device_type", "device_model"]
SIMULATABLE_FEATURES = {
    "hour", "banner_pos", "device_type", "device_model",
    "day_of_week", "hour_of_day", "app_category"
}

DEFAULT_ROW = {
    "hour":         1300,
    "banner_pos":     0,
    "device_type":    1,
    "device_model": "model_abc"
}

_booster = None
def _load_model():
    global _booster
    if _booster is None:
        _booster = lgb.Booster(model_file="/databricks/driver/ctr-predictor-module/model/ctr_model.txt")


# ————————————————————————————————
# 2) your existing SQL→CTR logic
# ————————————————————————————————
def run_sql_to_ctr_predictions(sql_query: str, spark_df) -> pd.DataFrame:
    spark_df.createOrReplaceTempView("ctr_df")
    clean_sql = sql_query.replace("__THIS__", "ctr_df")
    pdf = spark_df.sparkSession.sql(clean_sql).toPandas()

    # remember what the user asked for
    request_cols = list(pdf.columns)

    # if no rows, fall back to defaults
    if pdf.empty:
        pdf = pd.DataFrame([DEFAULT_ROW.copy()])
        request_cols = list(DEFAULT_ROW.keys())

    # drop any stale predictions
    pdf = pdf.drop(columns=[c for c in pdf.columns if c.lower()=="predicted_ctr"],
                   errors="ignore")
    request_cols = [c for c in request_cols if c.lower()!="predicted_ctr"]

    # ensure required columns exist
    for col, default in DEFAULT_ROW.items():
        if col not in pdf.columns:
            pdf[col] = default
        else:
            pdf[col] = pdf[col].fillna(default)

    # load and predict
    _load_model()

    # case A: we have everything → direct predict
    if all(col in pdf.columns for col in REQUIRED_COLUMNS):
        feats = create_features(pdf)
        pdf["predicted_ctr"] = _booster.predict(feats)
        return pdf[request_cols + ["predicted_ctr"]]

    # case B: exactly one simulatable column → sweep via mode
    sim_cols = [c for c in pdf.columns if c in SIMULATABLE_FEATURES]
    if len(sim_cols) == 1:
        sim_col = sim_cols[0]
        mode_row = spark_df.toPandas().mode(numeric_only=False).iloc[0].to_dict()
        out = []
        for _, row in pdf.iterrows():
            sim = mode_row.copy()
            sim[sim_col] = row[sim_col]
            feat = create_features(pd.DataFrame([sim]))
            p = _booster.predict(feat)[0]
            d = row.to_dict()
            d["predicted_ctr"] = p
            out.append(d)
        df_out = pd.DataFrame(out)
        return df_out[request_cols + ["predicted_ctr"]]

    # case C: nothing usable → one global prediction
    mode_row = spark_df.toPandas().mode(numeric_only=False).iloc[0].to_dict()
    feat = create_features(pd.DataFrame([mode_row]))
    p = _booster.predict(feat)[0]
    pdf["predicted_ctr"] = p
    return pdf[request_cols + ["predicted_ctr"]]


# ————————————————————————————————
# 3) UPDATED: take a `query` variable from your notebook
# ————————————————————————————————
def predict_from_question(query: str, spark_df) -> pd.DataFrame:
    """
    Pass a notebook variable `query` (e.g. query = "What is the predicted CTR for different device models?")
    """
    # load your key from env (set this in your notebook)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in the environment")

    system = """
You are a Spark SQL expert.
Translate the user's question about click-through-rate into a valid Spark SQL query
against the table `__THIS__`. Do not wrap the SQL in backticks.
"""
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content": system},
            {"role":"user",   "content": query}
        ],
        temperature=0,
        max_tokens=256
    )

    sql = resp.choices[0].message.content.strip()
    return run_sql_to_ctr_predictions(sql, spark_df)


__all__ = ["run_sql_to_ctr_predictions", "predict_from_question"]

# main_pipeline.py

import pandas as pd
import lightgbm as lgb
from feature_creator import create_features

import openai
import os

# ——— OPENAI SETUP ———
# Read your key from the environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable")
client = openai.OpenAI(api_key=api_key)

# ——— MODEL CONFIG ———
REQUIRED_COLUMNS     = ["hour", "banner_pos", "device_type", "device_model"]
SIMULATABLE_FEATURES = {
    "hour", "banner_pos", "device_type", "device_model",
    "day_of_week", "hour_of_day", "app_category"
}
DEFAULT_ROW = {
    "hour":        1300,
    "banner_pos":    0,
    "device_type":   1,
    "device_model": "model_abc"
}

_booster = None
def _load_model():
    global _booster
    if _booster is None:
        _booster = lgb.Booster(model_file="model/ctr_model.txt")

# ——— STEP 1: NL → SQL ———
def get_sql_query(spark_df, question: str) -> str:
    # register view (so we can introspect schema)
    spark_df.createOrReplaceTempView("ctr_df")

    cols = [f.name for f in spark_df.schema.fields]
    schema_hint = "The table `ctr_df` has columns:\n- " + "\n- ".join(cols)

    system_prompt = f"""
You are a Spark SQL expert.
Translate the user's natural-language question into a valid Spark SQL query over the table __THIS__.

Question:
{question}

Schema:
{schema_hint}

Do NOT wrap your answer in backticks.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content": system_prompt},
            {"role":"user",   "content": question}
        ]
    )
    return resp.choices[0].message.content

# ——— STEP 2: RUN & PREDICT ———
def run_sql_to_ctr_predictions(sql_query: str, spark_df) -> pd.DataFrame:
    spark_df.createOrReplaceTempView("ctr_df")
    clean_sql = sql_query.replace("__THIS__", "ctr_df")
    pdf = spark_df.sparkSession.sql(clean_sql).toPandas()

    # if no results, drop in a single default row
    if pdf.empty:
        pdf = pd.DataFrame([DEFAULT_ROW.copy()])

    # drop any old predicted_ctr
    pdf = pdf.drop(columns=[c for c in pdf.columns if c.lower()=="predicted_ctr"],
                   errors="ignore")

    # fill in missing required-cols
    for col, default in DEFAULT_ROW.items():
        if col not in pdf.columns:
            pdf[col] = default
        else:
            pdf[col] = pdf[col].fillna(default)

    _load_model()

    # a) if all required columns present → direct per-row predict
    if all(c in pdf.columns for c in REQUIRED_COLUMNS):
        feats = create_features(pdf)
        pdf["predicted_ctr"] = _booster.predict(feats)
        return pdf

    # b) if exactly one simulatable col present → sweep that “what-if”
    sim_cols = [c for c in pdf.columns if c in SIMULATABLE_FEATURES]
    if len(sim_cols) == 1:
        sim_col = sim_cols[0]
        mode_row = spark_df.toPandas().mode(numeric_only=False).iloc[0].to_dict()
        out = []
        for _, row in pdf.iterrows():
            inp = mode_row.copy()
            inp[sim_col] = row[sim_col]
            feats = create_features(pd.DataFrame([inp]))
            pred  = _booster.predict(feats)[0]
            d = row.to_dict(); d["predicted_ctr"] = pred
            out.append(d)
        return pd.DataFrame(out)

    # c) fallback: one global prediction on the “mode” row
    mode_row = spark_df.toPandas().mode(numeric_only=False).iloc[0].to_dict()
    feats = create_features(pd.DataFrame([mode_row]))
    pred  = _booster.predict(feats)[0]
    pdf["predicted_ctr"] = pred
    return pdf

# ——— CONVENIENCE WRAPPER ———
def predict_from_question(question: str, spark_df) -> pd.DataFrame:
    """
    1) turn NL → SQL
    2) run SQL + predict
    """
    sql = get_sql_query(spark_df, question)
    return run_sql_to_ctr_predictions(sql, spark_df)

#!/usr/bin/env python3
"""
Load CSVs (or Excel if needed) and prepare exports for Power BI.
- Reads CSV(s) from data/processed/ and data/predictions/
- Cleans/normalizes columns to match the DB schema
- Writes prepared CSVs into data/powerbi/
- Optionally loads into PostgreSQL if DATABASE_URL env var is set

Usage:
  python scripts/load_data_for_powerbi.py
  # or provide paths:
  python scripts/load_data_for_powerbi.py --predictions data/predictions/predictions.csv --engineered data/processed/churn_engineered.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Optional DB
import os
DATABASE_URL = os.getenv("DATABASE_URL")  # e.g. postgresql://user:pass@host:5432/churn_analytics
try:
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except Exception:
    SQLALCHEMY_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POWERBI_DIR = PROJECT_ROOT / "data" / "powerbi"
POWERBI_DIR.mkdir(parents=True, exist_ok=True)


def safe_read(path: Path) -> pd.DataFrame:
    """Read CSV or Excel safely into DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() in [".csv"]:
        return pd.read_csv(path)
    if path.suffix.lower() in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    # try csv by default
    return pd.read_csv(path)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names consistent (strip, remove spaces)."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def prepare_predictions_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename and prepare predictions dataframe to match DB / Power BI schema."""
    df = normalize_column_names(df)

    # mapping common names to target names
    column_mapping = {
        "customerID": "customer_id",
        "customer_id": "customer_id",
        "gender": "gender",
        "SeniorCitizen": "senior_citizen",
        "Senior Citizen": "senior_citizen",
        "Partner": "partner",
        "Dependents": "dependents",
        "tenure": "tenure",
        "PhoneService": "phone_service",
        "MultipleLines": "multiple_lines",
        "InternetService": "internet_service",
        "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection",
        "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv",
        "StreamingMovies": "streaming_movies",
        "Contract": "contract",
        "PaperlessBilling": "paperless_billing",
        "PaymentMethod": "payment_method",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "Churn": "churn_actual",
        "churn": "churn_actual",
        "churn_actual": "churn_actual",
    }

    # rename what's present
    rename_map = {k: v for k, v in column_mapping.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    # Clean TotalCharges (string -> numeric) safely
    if "total_charges" in df.columns:
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    else:
        # if not present, try to create from tenure * monthly_charges as fallback
        if "monthly_charges" in df.columns and "tenure" in df.columns:
            df["total_charges"] = df["monthly_charges"].astype(float) * df["tenure"].astype(float)
        else:
            df["total_charges"] = np.nan

    # Ensure monthly_charges numeric
    if "monthly_charges" in df.columns:
        df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="coerce")
    else:
        df["monthly_charges"] = 0.0

    # churn_prediction column: if not present derive from churn_actual
    if "churn_prediction" not in df.columns:
        if "churn_actual" in df.columns:
            df["churn_prediction"] = (df["churn_actual"].astype(str).str.lower().isin(["yes", "1", "true"])).astype(int)
        else:
            df["churn_prediction"] = 0

    # churn_probability: if not present, create placeholder between 0.1 and 0.9
    if "churn_probability" not in df.columns:
        rng = np.random.default_rng(42)
        df["churn_probability"] = rng.uniform(0.1, 0.9, size=len(df))

    # risk_level if not present
    if "risk_level" not in df.columns:
        df["risk_level"] = pd.cut(
            df["churn_probability"],
            bins=[-1, 0.4, 0.7, 1.0],
            labels=["Low", "Medium", "High"]
        ).astype(str)

    # convert senior flag to Yes/No
    if "senior_citizen" in df.columns:
        df["senior_citizen"] = df["senior_citizen"].map({0: "No", 1: "Yes", "0": "No", "1": "Yes"}).fillna(df["senior_citizen"])

    # keep and order columns relevant to Power BI + DB
    final_cols = [
        "customer_id", "gender", "senior_citizen", "partner", "dependents", "tenure",
        "phone_service", "multiple_lines", "internet_service", "online_security", "online_backup",
        "device_protection", "tech_support", "streaming_tv", "streaming_movies",
        "contract", "paperless_billing", "payment_method",
        "monthly_charges", "total_charges",
        "churn_actual", "churn_prediction", "churn_probability", "risk_level"
    ]
    # add any missing final columns to DF filled with NaN/0
    for col in final_cols:
        if col not in df.columns:
            df[col] = np.nan if df.shape[0] > 0 else []

    # return only final columns (plus any extras)
    ordered = [c for c in final_cols if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras]


def export_powerbi_csv(df: pd.DataFrame, filename: str):
    out_path = POWERBI_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"✅ Exported Power BI CSV: {out_path}")


def load_to_postgres(df: pd.DataFrame, table_name: str):
    if not SQLALCHEMY_AVAILABLE:
        print("⚠️  SQLAlchemy not available; skipping DB load. Install sqlalchemy and psycopg2-binary to enable.")
        return False
    if not DATABASE_URL:
        print("⚠️  DATABASE_URL env var not set; skipping DB load.")
        return False
    engine = create_engine(DATABASE_URL)
    try:
        df.to_sql(table_name, engine, if_exists="replace", index=False, method="multi", chunksize=1000)
        print(f"✅ Loaded {len(df)} rows into table `{table_name}`")
        return True
    except Exception as e:
        print(f"❌ Error loading to DB: {e}")
        return False


def main(args):
    # default file locations (prefer CSVs)
    default_predictions = PROJECT_ROOT / "data" / "predictions" / "predictions.csv"
    default_engineered = PROJECT_ROOT / "data" / "processed" / "churn_engineered.csv"
    default_raw = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

    preds_path = Path(args.predictions) if args.predictions else default_predictions
    eng_path = Path(args.engineered) if args.engineered else default_engineered
    raw_path = Path(args.raw) if args.raw else default_raw

    # Choose which file to use (predictions preferred)
    if preds_path.exists():
        print(f"Reading predictions file: {preds_path}")
        df_preds = safe_read(preds_path)
    elif eng_path.exists():
        print(f"No predictions found, reading engineered file: {eng_path}")
        df_preds = safe_read(eng_path)
    elif raw_path.exists():
        print(f"No processed files found, reading raw file: {raw_path}")
        df_preds = safe_read(raw_path)
    else:
        raise FileNotFoundError("No input CSV found. Place files under data/predictions/ or data/processed/")

    df_prepared = prepare_predictions_df(df_preds)

    # Export for Power BI
    export_powerbi_csv(df_prepared, "predictions_powerbi.csv")

    # Optionally create summary CSVs (quick metrics)
    summary = {
        "total_customers": len(df_prepared),
        "churners": int(df_prepared["churn_prediction"].sum()),
        "churn_rate_pct": float(df_prepared["churn_prediction"].sum() / max(1, len(df_prepared)) * 100),
        "avg_churn_probability": float(df_prepared["churn_probability"].mean())
    }
    pd.DataFrame([summary]).to_csv(POWERBI_DIR / "summary_metrics.csv", index=False)
    print(f"✅ Exported summary_metrics.csv")

    # Contract analysis
    contract_analysis = df_prepared.groupby("contract").agg(
        customer_count=("customer_id", "count"),
        churners=("churn_prediction", "sum"),
        avg_churn_prob=("churn_probability", "mean"),
        avg_monthly_charges=("monthly_charges", "mean"),
        avg_tenure=("tenure", "mean")
    ).reset_index()
    contract_analysis["churn_rate_pct"] = (contract_analysis["churners"] / contract_analysis["customer_count"]) * 100
    contract_analysis.to_csv(POWERBI_DIR / "contract_analysis.csv", index=False)
    print("✅ Exported contract_analysis.csv")

    # Internet analysis
    internet_analysis = df_prepared.groupby("internet_service").agg(
        customer_count=("customer_id", "count"),
        churners=("churn_prediction", "sum"),
        avg_churn_prob=("churn_probability", "mean")
    ).reset_index()
    internet_analysis["churn_rate_pct"] = (internet_analysis["churners"] / internet_analysis["customer_count"]) * 100
    internet_analysis.to_csv(POWERBI_DIR / "internet_analysis.csv", index=False)
    print("✅ Exported internet_analysis.csv")

    # Tenure bins
    bins = [0, 12, 24, 48, 72, 999]
    labels = ["0-12m", "12-24m", "24-48m", "48-72m", "72m+"]
    df_prepared["tenure_group"] = pd.cut(df_prepared["tenure"].fillna(0).astype(int), bins=bins, labels=labels, right=False)
    tenure_analysis = df_prepared.groupby("tenure_group").agg(
        customer_count=("customer_id", "count"),
        churners=("churn_prediction", "sum"),
        avg_churn_prob=("churn_probability", "mean")
    ).reset_index()
    tenure_analysis["churn_rate_pct"] = (tenure_analysis["churners"] / tenure_analysis["customer_count"]) * 100
    tenure_analysis.to_csv(POWERBI_DIR / "tenure_analysis.csv", index=False)
    print("✅ Exported tenure_analysis.csv")

    # Monthly trends simple (by simulated month from tenure)
    monthly = []
    for m in range(1, 13):
        subset = df_prepared.loc[df_prepared["tenure"].fillna(0) >= m]
        monthly.append({
            "month": m,
            "active_customers": len(subset),
            "high_risk_customers": int((subset["risk_level"] == "High").sum()),
            "predicted_churners": int(subset["churn_prediction"].sum()),
            "avg_churn_probability": float(subset["churn_probability"].mean()) if len(subset) else 0.0,
            "revenue_at_risk": float(subset.loc[subset["risk_level"] == "High", "monthly_charges"].sum())
        })
    pd.DataFrame(monthly).to_csv(POWERBI_DIR / "monthly_trends.csv", index=False)
    print("✅ Exported monthly_trends.csv")

    # Optionally load to Postgres if env var provided
    if DATABASE_URL and SQLALCHEMY_AVAILABLE:
        print("➡️ DATABASE_URL set; attempting to load `predictions` table into Postgres")
        load_to_postgres(df_prepared, "predictions")
    elif DATABASE_URL and not SQLALCHEMY_AVAILABLE:
        print("⚠️ DATABASE_URL set but SQLAlchemy not installed. Skipping DB load.")

    print("\nAll Power BI files saved under:", POWERBI_DIR.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CSVs for Power BI and optionally load to Postgres")
    parser.add_argument("--predictions", help="Path to predictions CSV (default: data/predictions/predictions.csv)")
    parser.add_argument("--engineered", help="Path to engineered CSV (default: data/processed/churn_engineered.csv)")
    parser.add_argument("--raw", help="Path to raw CSV (default: data/raw/Telco-Customer-Churn.csv)")
    args = parser.parse_args()
    main(args)

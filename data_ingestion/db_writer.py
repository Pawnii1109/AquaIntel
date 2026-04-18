"""
data_ingestion/db_writer.py
============================
Saves ARGO data to MySQL and provides query utilities.

FIXES from original:
  1. 'time' column is now properly saved
  2. 'float_id' column added
  3. Added run_query() method needed by RAG pipeline
  4. Added get_stats() for dashboard KPIs
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from sqlalchemy import create_engine, text
from config.settings import DB_URL
import logging

logger = logging.getLogger(__name__)

# Shared engine (created once, reused)
engine = create_engine(DB_URL, pool_pre_ping=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize fetched ARGO data before DB insert.
    Handles: column renaming, type conversion, null dropping.
    """
    df = df.copy()

    # ── Remove units row if present (ERDDAP quirk) ────────────────────────────
    # The first data row sometimes contains units like "degrees_north"
    if len(df) > 0 and isinstance(df["latitude"].iloc[0], str):
        df = df.iloc[1:]

    # ── Rename columns ────────────────────────────────────────────────────────
    rename = {
        "temp":            "temperature",
        "psal":            "salinity",
        "pres":            "depth",
        "platform_number": "float_id",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # ── Numeric conversion ────────────────────────────────────────────────────
    for col in ["latitude", "longitude", "temperature", "salinity", "depth"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Time conversion ───────────────────────────────────────────────────────
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df["time"] = df["time"].dt.tz_localize(None)   # MySQL doesn't store TZ

    # ── Ensure float_id column exists ─────────────────────────────────────────
    if "float_id" not in df.columns:
        df["float_id"] = "UNKNOWN"

    # ── Ensure depth column exists ────────────────────────────────────────────
    if "depth" not in df.columns:
        df["depth"] = None

    # ── Drop rows with critical nulls ─────────────────────────────────────────
    df = df.dropna(subset=["latitude", "longitude", "temperature"])

    # ── Keep only DB columns ──────────────────────────────────────────────────
    db_cols = ["float_id", "latitude", "longitude", "temperature",
               "salinity", "depth", "time"]
    df = df[[c for c in db_cols if c in df.columns]]

    return df


def save_to_db(df: pd.DataFrame, table: str = "ocean_data"):
    """
    Save cleaned DataFrame to MySQL in fast batch inserts.

    Args:
        df:    Cleaned DataFrame
        table: Target MySQL table name (default: ocean_data)
    """
    print(f"⏳ Inserting {len(df):,} rows into '{table}'...")
    try:
        df.to_sql(
            table,
            engine,
            if_exists="append",
            index=False,
            chunksize=5000,      # batch size to avoid memory issues
            method="multi",      # faster multi-row INSERT
        )
        print(f"✅ Inserted {len(df):,} rows successfully!")
    except Exception as e:
        print(f"❌ Insert failed: {e}")
        raise


def run_query(sql: str) -> pd.DataFrame:
    """
    Execute a SELECT SQL query and return results as DataFrame.
    Used by the RAG pipeline to answer user questions.

    Args:
        sql: A SELECT SQL query string

    Returns:
        pandas DataFrame with query results
    """
    try:
        with engine.connect() as conn:
            return pd.read_sql_query(text(sql), conn)
    except Exception as e:
        logger.error(f"Query failed: {e}\nSQL: {sql}")
        raise


def get_stats() -> dict:
    """
    Return summary statistics from the database for the dashboard KPI cards.
    """
    sql = """
        SELECT
            COUNT(DISTINCT float_id)   AS total_floats,
            COUNT(*)                   AS total_records,
            ROUND(AVG(temperature), 2) AS avg_temperature,
            ROUND(AVG(salinity), 2)    AS avg_salinity,
            ROUND(MAX(depth), 0)       AS max_depth,
            MIN(time)                  AS earliest_date,
            MAX(time)                  AS latest_date
        FROM ocean_data
        WHERE temperature IS NOT NULL
    """
    try:
        df = run_query(sql)
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception as e:
        logger.error(f"get_stats failed: {e}")
        return {}


if __name__ == "__main__":
    # Full pipeline test: fetch → clean → save
    from data_ingestion.argo_fetcher import fetch_data

    print("📥 Fetching data from ERDDAP...")
    df = fetch_data(region="arabian_sea")

    if df.empty:
        print("❌ No data fetched. Check internet connection.")
    else:
        print("🧹 Cleaning data...")
        df = clean_data(df)
        print(f"📊 Rows to insert: {len(df)}")
        save_to_db(df)

        print("\n📈 Database stats:")
        stats = get_stats()
        for k, v in stats.items():
            print(f"   {k}: {v}")
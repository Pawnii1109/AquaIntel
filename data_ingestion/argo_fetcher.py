

import requests
import pandas as pd
from io import StringIO
import urllib3
import logging

# Suppress SSL warning cleanly (ERDDAP cert issue)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# ── Geographic presets ────────────────────────────────────────────────────────
REGIONS = {
    "default": {
        "lat_min": -10, "lat_max": 10,
        "lon_min": -10, "lon_max": 10,
    },
    "indian_ocean": {
        "lat_min": -30, "lat_max": 25,
        "lon_min":  40, "lon_max": 100,
    },
    "arabian_sea": {
        "lat_min":   5, "lat_max": 25,
        "lon_min":  55, "lon_max": 78,
    },
    "bay_of_bengal": {
        "lat_min":  5, "lat_max": 22,
        "lon_min": 80, "lon_max": 100,
    },
    "global": {
        "lat_min": -60, "lat_max": 60,
        "lon_min": -180, "lon_max": 180,
    },
}

ERDDAP_BASE = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv"


def build_url(region: str = "default", pres_max: float = 200.0) -> str:
    """Build ERDDAP query URL for given region and max pressure."""
    r = REGIONS.get(region, REGIONS["default"])
    url = (
        f"{ERDDAP_BASE}"
        f"?platform_number,latitude,longitude,time,temp,psal,pres"
        f"&latitude>={r['lat_min']}&latitude<={r['lat_max']}"
        f"&longitude>={r['lon_min']}&longitude<={r['lon_max']}"
        f"&pres<={pres_max}"
        f"&orderBy(\"time\")"
    )
    return url


def fetch_data(region: str = "default", pres_max: float = 200.0) -> pd.DataFrame:
    """
    Fetch ARGO data from ERDDAP for a given region.

    Args:
        region:   One of 'default', 'indian_ocean', 'arabian_sea',
                  'bay_of_bengal', 'global'
        pres_max: Max pressure/depth in dbar to fetch

    Returns:
        Clean DataFrame with columns:
        float_id, latitude, longitude, time, temperature, salinity, depth
    """
    url = build_url(region, pres_max)
    logger.info(f"Fetching ARGO data for region='{region}' from ERDDAP...")
    print(f"🌊 Fetching region: {region}")
    print(f"   URL: {url[:80]}...")

    try:
        response = requests.get(url, verify=False, timeout=60)
    except requests.exceptions.Timeout:
        print("❌ Request timed out. Try a smaller region.")
        return pd.DataFrame()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to ERDDAP. Check internet connection.")
        return pd.DataFrame()

    if response.status_code != 200:
        print(f"❌ ERDDAP returned status {response.status_code}")
        print(f"   Response: {response.text[:300]}")
        return pd.DataFrame()

    print("✅ Data fetched successfully")

    # ── Parse CSV ─────────────────────────────────────────────────────────────
    # ERDDAP returns 2-row header: row 0 = column names, row 1 = units
    try:
        df = pd.read_csv(StringIO(response.text), header=0, skiprows=[1])
    except Exception as e:
        print(f"❌ Failed to parse CSV: {e}")
        return pd.DataFrame()

    # ── Rename columns to our schema ──────────────────────────────────────────
    col_map = {
        "platform_number": "float_id",
        "temp":            "temperature",
        "psal":            "salinity",
        "pres":            "depth",      # pressure ≈ depth in meters
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # ── Convert types ─────────────────────────────────────────────────────────
    numeric_cols = ["latitude", "longitude", "temperature", "salinity", "depth"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        # Remove timezone info for MySQL compatibility
        df["time"] = df["time"].dt.tz_localize(None)

    # ── Drop critical nulls ───────────────────────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude", "temperature"])
    after = len(df)
    print(f"📊 Rows: {before} fetched → {after} after dropping nulls")

    return df


if __name__ == "__main__":
    # Quick test
    df = fetch_data(region="arabian_sea")
    if not df.empty:
        print("\nSample data:")
        print(df.head())
        print("\nColumns:", df.columns.tolist())
        print("Shape:", df.shape)

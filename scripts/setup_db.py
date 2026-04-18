"""
scripts/setup_db.py
====================
Run this ONCE to create the MySQL database schema.

FIXED: Added 'time' column which was missing in original version.

Usage:
    python scripts/setup_db.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy import create_engine, text
from config.settings import DB_URL

engine = create_engine(DB_URL)

with engine.connect() as conn:

    # ── Drop old table if re-running ──────────────────────────────────────────
    # Uncomment next line ONLY if you want to reset completely:
    # conn.execute(text("DROP TABLE IF EXISTS ocean_data;"))

    # ── Create table with ALL columns ────────────────────────────────────────
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS ocean_data (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            latitude    FLOAT,
            longitude   FLOAT,
            temperature FLOAT,
            salinity    FLOAT,
            depth       FLOAT,
            time        DATETIME,
            float_id    VARCHAR(50)
        );
    """))

    # ── Useful indexes for fast queries ──────────────────────────────────────
    # We use TRY because indexes may already exist on re-run
    try:
        conn.execute(text("CREATE INDEX idx_lat_lon ON ocean_data(latitude, longitude);"))
    except Exception:
        pass
    try:
        conn.execute(text("CREATE INDEX idx_time ON ocean_data(time);"))
    except Exception:
        pass

    conn.commit()
    print("✅ Table 'ocean_data' created (or already exists) with all columns.")
    print("   Columns: id, latitude, longitude, temperature, salinity, depth, time, float_id")
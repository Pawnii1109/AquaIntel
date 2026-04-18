"""
test_all.py
============
Verifies all FloatChat components before running the Streamlit app.

Usage:
    cd D:\mypyprogs\AquaIntel
    python test_all.py
"""

import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

results = []

def check(name, fn):
    try:
        msg = fn()
        results.append((name, True, msg or "OK"))
        print(f"  ✅ PASS  {name}" + (f" — {msg}" if msg else ""))
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  ❌ FAIL  {name}")
        print(f"           {e}")

print("\n" + "="*58)
print("  FloatChat — Component Test Suite  (Gemini + MySQL)")
print("="*58 + "\n")

# ── 1. Config ─────────────────────────────────────────────────
print("── 1. Config ─────────────────────────────────────────────")

def test_config():
    from config.settings import DB_URL, GEMINI_API_KEY
    assert DB_URL,         "DB_URL empty — check DB credentials in .env"
    assert GEMINI_API_KEY, "GEMINI_API_KEY empty — add it to .env"
    assert GEMINI_API_KEY != "your_gemini_api_key_here", \
        "GEMINI_API_KEY still has placeholder value — replace with real key"
    return f"DB_URL set, GEMINI_API_KEY set (len={len(GEMINI_API_KEY)})"

check("Config & .env", test_config)

# ── 2. MySQL ──────────────────────────────────────────────────
print("\n── 2. MySQL ──────────────────────────────────────────────")

def test_db_connect():
    from sqlalchemy import create_engine, text
    from config.settings import DB_URL
    engine = create_engine(DB_URL, pool_pre_ping=True)
    with engine.connect() as conn:
        r = conn.execute(text("SELECT 1")).fetchone()
    assert r[0] == 1
    return "Connected"

check("MySQL connection", test_db_connect)

def test_table():
    from sqlalchemy import create_engine, text
    from config.settings import DB_URL
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        r = conn.execute(text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema=DATABASE() AND table_name='ocean_data'"
        )).fetchone()
    assert r[0] == 1, "Table not found — run: python scripts/setup_db.py"
    return "Table ocean_data exists"

check("Table ocean_data", test_table)

def test_data():
    from data_ingestion.db_writer import run_query
    df = run_query("SELECT COUNT(*) AS cnt FROM ocean_data")
    cnt = int(df.iloc[0]["cnt"])
    assert cnt > 0, f"Table empty ({cnt} rows) — run: python data_ingestion/db_writer.py"
    return f"{cnt:,} rows"

check("Data in ocean_data", test_data)

def test_stats():
    from data_ingestion.db_writer import get_stats
    s = get_stats()
    assert s, "get_stats() returned empty"
    return f"floats={s.get('total_floats')}, avg_temp={s.get('avg_temperature')}°C"

check("get_stats()", test_stats)

# ── 3. ERDDAP ─────────────────────────────────────────────────
print("\n── 3. ERDDAP ─────────────────────────────────────────────")

def test_erddap():
    import requests, urllib3
    urllib3.disable_warnings()
    r = requests.get(
        "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv"
        "?latitude&latitude>=0&latitude<=1&longitude>=0&longitude<=1"
        "&pres<=10&orderBy(%22time%22)&.limit=1",
        verify=False, timeout=15
    )
    assert r.status_code == 200, f"Status {r.status_code}"
    return "ERDDAP reachable"

check("ERDDAP API", test_erddap)

# ── 4. Vector Store ───────────────────────────────────────────
print("\n── 4. Vector Store ───────────────────────────────────────")

def test_faiss():
    import faiss, numpy as np
    idx = faiss.IndexFlatIP(8)
    vecs = np.random.rand(3, 8).astype("float32")
    idx.add(vecs)
    assert idx.ntotal == 3
    return "FAISS working"

check("FAISS", test_faiss)

def test_sbert():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    e = m.encode(["test"])
    assert e.shape[1] == 384
    return f"dim={e.shape[1]}"

check("sentence-transformers", test_sbert)

def test_index_file():
    from config.settings import FAISS_INDEX_PATH, FAISS_META_PATH
    assert os.path.exists(FAISS_INDEX_PATH), \
        f"Not found: {FAISS_INDEX_PATH} — run: python vector_store/embedder.py"
    assert os.path.exists(FAISS_META_PATH)
    return FAISS_INDEX_PATH

check("FAISS index file", test_index_file)

def test_search():
    from vector_store.embedder import Embedder
    e = Embedder()
    e.load()
    r = e.search("Arabian Sea temperature", top_k=3)
    assert isinstance(r, list)
    return f"{len(r)} results"

check("Embedder search", test_search)

# ── 5. Gemini (NEW SDK) ───────────────────────────────────────
print("\n── 5. Gemini (google.genai SDK) ──────────────────────────")

def test_genai_import():
    from google import genai
    from google.genai import types
    return "google.genai imported"

check("google.genai import", test_genai_import)

def test_gemini_call():
    from google import genai
    from google.genai import types
    from config.settings import GEMINI_API_KEY
    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Reply with the single word: ocean",
        config=types.GenerateContentConfig(max_output_tokens=10),
    )
    assert resp.text.strip(), "Empty response"
    return f"Response: '{resp.text.strip()[:20]}'"

check("Gemini API call", test_gemini_call)

def test_text_to_sql():
    from llm_service.text_to_sql import TextToSQL
    t = TextToSQL()
    sql = t.convert("average temperature?")
    assert sql.upper().startswith("SELECT"), f"Not a SELECT: {sql[:60]}"
    assert "temperature" in sql.lower()
    return f"SQL: {sql[:60]}..."

check("Text-to-SQL", test_text_to_sql)

# ── 6. Charts ─────────────────────────────────────────────────
print("\n── 6. Charts ─────────────────────────────────────────────")

def test_charts():
    import pandas as pd
    from visualization.chart_engine import (
        make_line_chart, make_bar_chart, make_depth_profile, make_scatter_plot
    )
    df = pd.DataFrame({
        "month": ["2024-01","2024-02","2024-03"],
        "temperature": [27.1, 26.8, 27.5],
        "salinity": [35.1, 35.2, 35.0],
        "depth": [10, 50, 100],
    })
    figs = [
        make_line_chart(df, "month", "temperature", "T"),
        make_bar_chart(df, "month", "temperature", "T"),
        make_depth_profile(df, "T"),
        make_scatter_plot(df, "salinity", "temperature", "T"),
    ]
    assert all(f is not None for f in figs)
    return "4 chart types OK"

check("Chart engine", test_charts)

# ── Summary ───────────────────────────────────────────────────
total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed

print(f"\n{'='*58}")
print(f"  RESULTS: {passed}/{total} passed")
print(f"{'='*58}")

if failed == 0:
    print("\n  🎉 ALL TESTS PASSED!")
    print("  Launch app:  streamlit run frontend/app.py\n")
else:
    print(f"\n  ⚠️  {failed} test(s) failed:\n")
    for name, ok, msg in results:
        if not ok:
            print(f"  ❌ {name}")
            print(f"     {msg}\n")
    print("  Fix commands:")
    print("  ├─ No table?     python scripts/setup_db.py")
    print("  ├─ No data?      python data_ingestion/db_writer.py")
    print("  ├─ No index?     python vector_store/embedder.py")
    print("  └─ Gemini fail?  check GEMINI_API_KEY in .env")
    print()
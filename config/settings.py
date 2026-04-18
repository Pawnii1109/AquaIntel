"""
config/settings.py
==================
Central configuration for FloatChat.
All secrets read from .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── MySQL ──────────────────────────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = os.getenv("DB_PORT",     "3306")
DB_NAME     = os.getenv("DB_NAME",     "floatchat")
DB_USER     = os.getenv("DB_USER",     "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── Gemini ─────────────────────────────────────────────────────────────────────
# Get free key at: https://aistudio.google.com/apikey
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# NOTE: GEMINI_MODEL is now hardcoded in each llm_service file as "gemini-2.0-flash"
# The old "gemini-1.5-flash" model name is not supported in the new google.genai SDK

# ── FAISS ──────────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = "vector_store/faiss.index"
FAISS_META_PATH  = "vector_store/metadata.json"

# ── App ────────────────────────────────────────────────────────────────────────
APP_TITLE    = "FloatChat — ARGO Ocean AI"
APP_SUBTITLE = "SIH25040 | AI-Powered Ocean Data Intelligence"
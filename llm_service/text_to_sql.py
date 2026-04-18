"""
llm_service/text_to_sql.py
===========================
Converts natural language questions to MySQL SELECT queries using Gemini.

FIXED: Uses new google.genai SDK (not deprecated google.generativeai)
       Model: gemini-2.0-flash

USAGE:
    python llm_service/text_to_sql.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import re
import logging
from google import genai
from google.genai import types
from config.settings import GEMINI_API_KEY

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.0-flash"

# ── Table schema ──────────────────────────────────────────────────────────────
DB_SCHEMA = """
DATABASE: MySQL
TABLE: ocean_data

COLUMNS:
  id          INT           -- auto primary key
  float_id    VARCHAR(50)   -- ARGO float WMO ID
  time        DATETIME      -- measurement timestamp
  latitude    FLOAT         -- decimal degrees, positive=North
  longitude   FLOAT         -- decimal degrees, positive=East
  temperature FLOAT         -- temperature in Celsius
  salinity    FLOAT         -- salinity in PSU
  depth       FLOAT         -- depth in dbar (~meters)

GEOGRAPHIC REFERENCE:
  Indian Ocean:    latitude BETWEEN -30 AND 30,  longitude BETWEEN 20 AND 120
  Arabian Sea:     latitude BETWEEN 5 AND 25,    longitude BETWEEN 55 AND 78
  Bay of Bengal:   latitude BETWEEN 5 AND 22,    longitude BETWEEN 80 AND 100
  Southern Ocean:  latitude BETWEEN -60 AND -40
  Atlantic Ocean:  latitude BETWEEN -60 AND 70,  longitude BETWEEN -80 AND 20
"""

SYSTEM_PROMPT = f"""
You are an expert MySQL ocean science assistant for FloatChat.
Convert natural language questions about ARGO ocean float data into MySQL SELECT queries.

{DB_SCHEMA}

RULES:
1. Output ONLY the raw SQL — no explanation, no markdown, no backticks.
2. Only SELECT queries — never INSERT, UPDATE, DELETE, DROP, ALTER, CREATE.
3. Add LIMIT 5000 for row queries (not aggregations).
4. MySQL date functions: DATE_FORMAT(time, '%Y-%m') for month, YEAR(time) for year.
5. Use ROUND(value, 3) for floats.
6. Use geographic lat/lon ranges for named regions.
7. If unanswerable: SELECT 'CANNOT_ANSWER' AS error;

EXAMPLES:
Q: Average temperature in Arabian Sea?
A: SELECT ROUND(AVG(temperature), 3) AS avg_temperature FROM ocean_data WHERE latitude BETWEEN 5 AND 25 AND longitude BETWEEN 55 AND 78 AND temperature IS NOT NULL;

Q: Temperature trend by month?
A: SELECT DATE_FORMAT(time, '%Y-%m') AS month, ROUND(AVG(temperature), 3) AS avg_temperature FROM ocean_data WHERE temperature IS NOT NULL GROUP BY month ORDER BY month;

Q: Which float has most records?
A: SELECT float_id, COUNT(*) AS record_count FROM ocean_data GROUP BY float_id ORDER BY record_count DESC LIMIT 10;

Q: Salinity deeper than 100m?
A: SELECT float_id, time, latitude, longitude, depth, ROUND(salinity, 3) AS salinity FROM ocean_data WHERE depth > 100 AND salinity IS NOT NULL ORDER BY time DESC LIMIT 5000;
"""


class TextToSQL:
    """Converts natural language to MySQL using google.genai SDK."""

    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model  = GEMINI_MODEL
        print(f"✅ TextToSQL ready — model: {self.model}")

    def convert(self, question: str) -> str:
        full_prompt = f"{SYSTEM_PROMPT}\n\nQ: {question}\nA:"
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=400,
                ),
            )
            sql = response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            print(f"❌ Gemini API error: {e}")
            return "SELECT 'API_ERROR' AS error;"

        return self._sanitize(sql)

    def _sanitize(self, sql: str) -> str:
        sql = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
        sql = sql.strip("`").strip()

        forbidden = re.compile(
            r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE)\b",
            re.IGNORECASE
        )
        if forbidden.search(sql):
            return "SELECT 'BLOCKED: Only SELECT allowed' AS error;"

        if not sql.strip().upper().startswith("SELECT"):
            return "SELECT 'INVALID_QUERY' AS error;"

        return sql


if __name__ == "__main__":
    t2sql = TextToSQL()
    print()
    questions = [
        "What is the average temperature in the Arabian Sea?",
        "Which float has the most records?",
        "Show me temperature trend by month",
        "What is the salinity at depth greater than 100m?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"SQL: {t2sql.convert(q)}")
        print()
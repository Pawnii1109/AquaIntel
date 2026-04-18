

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import logging
import pandas as pd
from google import genai
from google.genai import types

from llm_service.text_to_sql import TextToSQL
from vector_store.embedder import Embedder
from data_ingestion.db_writer import run_query
from config.settings import GEMINI_API_KEY

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.0-flash"

ANSWER_PROMPT = """
You are FloatChat, an expert AI assistant for ARGO ocean float data.

Given a user question, retrieved profile context, and SQL query results,
you must respond in this EXACT JSON format (no extra text, no markdown):
{
  "answer": "2-4 sentence answer using real numbers from the data",
  "chart_type": "line_chart",
  "chart_title": "Descriptive chart title",
  "chart_x": "exact_column_name",
  "chart_y": "exact_column_name",
  "insight": "One interesting oceanographic fact explaining the data"
}

CHART TYPE — pick the single best one:
  line_chart    → time trends (x=month, y=temperature or salinity)
  bar_chart     → comparing floats or categories (x=float_id, y=record_count)
  scatter_map   → geographic positions (needs latitude+longitude columns)
  depth_profile → temperature or salinity vs depth
  scatter_plot  → any two numeric columns
  none          → when no chart adds value

IMPORTANT: chart_x and chart_y must be EXACT column names from the SQL results.
"""


class RAGPipeline:
    """Full FloatChat query pipeline using new google.genai SDK."""

    def __init__(self):
        print("🔧 Initializing RAG Pipeline...")

        # Vector store
        self.embedder = Embedder()
        try:
            self.embedder.load()
        except FileNotFoundError:
            print("⚠️  FAISS index not found. Run: python vector_store/embedder.py")

        # Text-to-SQL
        self.t2sql = TextToSQL()

        # Gemini client (new SDK)
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model  = GEMINI_MODEL

        print("✅ RAG Pipeline ready.")

    def query(self, question: str) -> dict:
        """
        Full pipeline: question → retrieve → SQL → execute → answer + chart.

        Returns dict with: answer, chart_type, chart_title, chart_x, chart_y,
                           insight, sql, data (DataFrame), context, error
        """
        result = {
            "answer": "", "chart_type": "none",
            "chart_title": "", "chart_x": "", "chart_y": "",
            "insight": "", "sql": "", "data": None,
            "context": [], "error": None,
        }

        print(f"\n🔍 Query: {question}")

        # ── Step 1: Semantic Retrieval ─────────────────────────────────────────
        try:
            context_chunks = self.embedder.search(question, top_k=5)
            result["context"] = context_chunks
            context_text = "\n".join(c.get("text", "") for c in context_chunks)
            print(f"   📚 {len(context_chunks)} context chunks retrieved")
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            context_text = ""

        # ── Step 2: Text-to-SQL ────────────────────────────────────────────────
        try:
            sql = self.t2sql.convert(question)
            result["sql"] = sql
            print(f"   🔧 SQL: {sql[:80]}...")
        except Exception as e:
            result["error"] = f"SQL generation failed: {e}"
            return result

        # ── Step 3: Execute SQL ────────────────────────────────────────────────
        try:
            df = run_query(sql)
            result["data"] = df
            data_preview = df.head(20).to_string(index=False) if not df.empty else "No data returned."
            print(f"   📊 {len(df)} rows returned")
        except Exception as e:
            logger.warning(f"SQL execution failed: {e}")
            data_preview = f"Query error: {e}"
            result["error"] = str(e)
            df = pd.DataFrame()

        # ── Step 4: Synthesize Answer ──────────────────────────────────────────
        full_prompt = f"""
{ANSWER_PROMPT}

User question: {question}

Retrieved ocean profile context:
{context_text if context_text else "No context available."}

SQL query used:
{sql}

Query results (first 20 rows):
{data_preview}

Respond ONLY with the JSON object.
"""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=600,
                ),
            )
            raw = response.text.strip()

            # Strip markdown fences if Gemini added them
            raw = raw.replace("```json", "").replace("```", "").strip()

            parsed = json.loads(raw)
            result.update(parsed)
            print(f"   ✅ Answer: {result['answer'][:60]}...")

        except json.JSONDecodeError:
            logger.warning(f"JSON parse failed. Raw response: {raw[:200]}")
            result["answer"] = self._fallback_answer(df)
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            result["answer"] = f"Data found but could not generate answer: {e}"

        return result

    def _fallback_answer(self, df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "No data found for this question."
        return (f"Found {len(df)} records. "
                f"Columns: {list(df.columns)}. "
                f"Sample: {df.head(2).to_dict(orient='records')}")


if __name__ == "__main__":
    pipeline = RAGPipeline()

    questions = [
        "What is the average temperature in the Arabian Sea?",
        "Show me temperature trend by month",
        "Which float collected the most data?",
    ]

    for q in questions:
        print(f"\n{'='*55}")
        result = pipeline.query(q)
        print(f"Answer:     {result['answer']}")
        print(f"Chart type: {result['chart_type']}")
        print(f"Chart x/y:  {result['chart_x']} / {result['chart_y']}")
        if result.get("error"):
            print(f"Error:      {result['error']}")

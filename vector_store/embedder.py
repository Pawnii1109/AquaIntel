

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from data_ingestion.db_writer import run_query
from config.settings import FAISS_INDEX_PATH, FAISS_META_PATH

logger = logging.getLogger(__name__)

# ── Embedding model ───────────────────────────────────────────────────────────
# all-MiniLM-L6-v2:  384 dimensions, very fast, great quality
# Downloads ~90MB on first run, then cached locally
MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """
    Manages the FAISS vector index for ARGO ocean profile metadata.

    Attributes:
        model:    SentenceTransformer instance (runs locally, no API key needed)
        index:    FAISS IndexFlatIP (dot product = cosine similarity after L2 norm)
        metadata: list of dicts matching each FAISS vector
    """

    def __init__(self):
        print("🧠 Loading embedding model (downloads once ~90MB)...")
        self.model    = SentenceTransformer(MODEL_NAME)
        self.index    = None
        self.metadata = []
        print("✅ Embedding model ready.")

    # ── Normalization for cosine similarity ───────────────────────────────────
    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / (norms + 1e-9)

    # ── Build from MySQL ───────────────────────────────────────────────────────
    def build_from_db(self):
        """
        Pull ocean data from MySQL, generate text descriptions,
        and build the FAISS index. Run this after loading data into DB.
        """
        print("📥 Pulling profile summaries from MySQL...")

        # Get one row per (float_id, date) combination = one "profile"
        sql = """
            SELECT
                float_id,
                DATE(time)                   AS profile_date,
                ROUND(AVG(latitude),  4)     AS latitude,
                ROUND(AVG(longitude), 4)     AS longitude,
                ROUND(AVG(temperature), 3)   AS avg_temp,
                ROUND(MIN(temperature), 3)   AS min_temp,
                ROUND(MAX(temperature), 3)   AS max_temp,
                ROUND(AVG(salinity), 3)      AS avg_sal,
                ROUND(MAX(depth), 0)         AS max_depth,
                COUNT(*)                     AS n_levels
            FROM ocean_data
            WHERE temperature IS NOT NULL
            GROUP BY float_id, DATE(time)
            ORDER BY profile_date DESC
            LIMIT 5000
        """

        df = run_query(sql)

        if df.empty:
            print("❌ No data in DB. Run db_writer.py first.")
            return

        print(f"   Found {len(df):,} unique profiles")

        # ── Generate text descriptions ────────────────────────────────────────
        chunks = []
        for _, row in df.iterrows():
            float_id     = str(row.get("float_id", "UNKNOWN")).strip()
            date         = str(row.get("profile_date", ""))
            lat          = float(row.get("latitude",  0))
            lon          = float(row.get("longitude", 0))
            avg_temp     = row.get("avg_temp",  "N/A")
            min_temp     = row.get("min_temp",  "N/A")
            max_temp     = row.get("max_temp",  "N/A")
            avg_sal      = row.get("avg_sal",   "N/A")
            max_depth    = row.get("max_depth", "N/A")

            # Determine approximate ocean region
            region = _get_region(lat, lon)

            text = (
                f"ARGO float {float_id} profile measured on {date} "
                f"in the {region} at latitude {lat:.2f}°, longitude {lon:.2f}°. "
                f"Temperature: avg {avg_temp}°C, min {min_temp}°C, max {max_temp}°C. "
                f"Average salinity: {avg_sal} PSU. "
                f"Maximum depth recorded: {max_depth} dbar."
            )

            chunks.append({
                "float_id":  float_id,
                "date":      date,
                "latitude":  lat,
                "longitude": lon,
                "region":    region,
                "avg_temp":  avg_temp,
                "avg_sal":   avg_sal,
                "text":      text,
            })

        # ── Embed all texts ───────────────────────────────────────────────────
        print(f"🔢 Embedding {len(chunks)} texts (may take ~30 seconds)...")
        texts      = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=64)
        embeddings = self._normalize(embeddings.astype("float32"))

        # ── Build FAISS index ─────────────────────────────────────────────────
        dim        = embeddings.shape[1]   # 384 for MiniLM
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata = chunks

        print(f"✅ FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    # ── Save to disk ───────────────────────────────────────────────────────────
    def save(self, index_path: str = FAISS_INDEX_PATH,
             meta_path: str = FAISS_META_PATH):
        """Save the FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, default=str, indent=2)
        print(f"💾 Saved index → {index_path}")
        print(f"💾 Saved metadata → {meta_path}")

    # ── Load from disk ─────────────────────────────────────────────────────────
    def load(self, index_path: str = FAISS_INDEX_PATH,
             meta_path: str = FAISS_META_PATH):
        """Load a previously saved FAISS index from disk."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run: python vector_store/embedder.py"
            )
        self.index = faiss.read_index(index_path)
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
        print(f"✅ Loaded FAISS index: {self.index.ntotal} vectors")

    # ── Semantic search ────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Find the most relevant ocean profiles for a natural language query.

        Args:
            query:  User's question e.g. "temperature in Arabian Sea"
            top_k:  Number of matching profiles to return

        Returns:
            List of metadata dicts sorted by relevance, each with 'score' key
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index empty. Call build_from_db() first.")
            return []

        q_emb = self.model.encode([query])
        q_emb = self._normalize(q_emb.astype("float32"))

        scores, indices = self.index.search(q_emb, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            results.append(entry)

        return results


# ── Helper: map lat/lon to ocean region name ───────────────────────────────────
def _get_region(lat: float, lon: float) -> str:
    """Return approximate ocean region name from lat/lon."""
    if 5 <= lat <= 25 and 55 <= lon <= 78:
        return "Arabian Sea"
    elif 5 <= lat <= 22 and 80 <= lon <= 100:
        return "Bay of Bengal"
    elif -30 <= lat <= 30 and 20 <= lon <= 120:
        return "Indian Ocean"
    elif -60 <= lat <= -40:
        return "Southern Ocean"
    elif -60 <= lat <= 60 and (120 <= lon or lon <= -70):
        return "Pacific Ocean"
    elif -60 <= lat <= 70 and -80 <= lon <= 20:
        return "Atlantic Ocean"
    elif 30 <= lat <= 46 and -6 <= lon <= 42:
        return "Mediterranean Sea"
    else:
        return "Open Ocean"


# ── Entry point: build and save index ────────────────────────────────────────
if __name__ == "__main__":
    emb = Embedder()
    emb.build_from_db()

    if emb.index and emb.index.ntotal > 0:
        emb.save()
        print("\n🔍 Test search: 'temperature in Arabian Sea'")
        results = emb.search("temperature in Arabian Sea", top_k=3)
        for r in results:
            print(f"   Score={r['score']:.3f} | {r['text'][:100]}...")
    else:
        print("❌ Index is empty — check that ocean_data table has data.")

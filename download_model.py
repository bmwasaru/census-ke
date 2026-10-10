from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DESTINATION = Path("models/bge-small-en-v1.5")
DESTINATION.parent.mkdir(parents=True, exist_ok=True)
print("Downloading BGE model...")
model = SentenceTransformer(MODEL_NAME)
model.save(str(DESTINATION))
print(f"Model saved locally to {DESTINATION}")

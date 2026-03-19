# config_embedding.py
import os

# -------------------------------
# ✏️ MANUALLY SELECT ONE MODEL HERE:
# Uncomment exactly one line to activate a model.
# Each line includes the reason why you'd use it.

DEFAULT_MODEL = "bge_small"       # ✅ Best trade-off between speed and quality. Great for general-purpose vector search and fast inference.
# DEFAULT_MODEL = "miniLM"          # ⚡ Fastest model. Ideal for prototypes, low-resource environments, or real-time retrieval where latency matters.
# DEFAULT_MODEL = "multiqa_mpnet"   # 🧠 Fine-tuned for QA-style search. Excellent for retrieval tasks where user queries are full sentences/questions.
# DEFAULT_MODEL = "all_mpnet"       # 🔍 High semantic quality. Great all-rounder for diverse query types, but slower than MiniLM.
# DEFAULT_MODEL = "bge_large"       # 🥇 Superior quality. Use when quality is top priority and you have compute/GPU to support it.
# DEFAULT_MODEL = "e5_large"          # 🏆 Highest quality. Best for long documents and highly nuanced semantic retrieval. Slowest but most powerful.
# -------------------------------


# Models ordered by ranking (real-time performance balanced with quality)
MODEL_CHOICES = {
    # Best balance of speed and quality
    "bge_small": {
        "name": "BAAI/bge-small-en-v1.5",
        "dimension": 384,
        "metric": "IP",
        "rank": 1
    },
    # Fastest but lower quality
    "miniLM": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "metric": "IP",
        "rank": 2
    },
    # Good quality-speed middle ground
    "multiqa_mpnet": {
        "name": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "dimension": 768,
        "metric": "IP",
        "rank": 3
    },
    # Higher quality but slower
    "all_mpnet": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "dimension": 768,
        "metric": "IP", 
        "rank": 4
    },
    # Best quality but significantly slower queries
    "bge_large": {
        "name": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "metric": "IP",
        "rank": 5
    },
    # Highest quality but slowest
    "e5_large": {
        "name": "intfloat/e5-large-v2",
        "dimension": 1024,
        "metric": "IP",
        "rank": 6
    }
}

# Use environment variable if set; fallback to default
SELECTED_MODEL_ALIAS = os.getenv("EMBED_MODEL_ALIAS", DEFAULT_MODEL)

# If an invalid model is specified, use the default
if SELECTED_MODEL_ALIAS not in MODEL_CHOICES:
    print(f"Warning: Unknown model alias '{SELECTED_MODEL_ALIAS}'. Using default: {DEFAULT_MODEL}")
    SELECTED_MODEL_ALIAS = DEFAULT_MODEL

# Extract all needed properties from the selected model configuration
EMBEDDING_MODEL_NAME = MODEL_CHOICES[SELECTED_MODEL_ALIAS]["name"]
EMBEDDING_DIMENSION = MODEL_CHOICES[SELECTED_MODEL_ALIAS]["dimension"]
SIMILARITY_METRIC = MODEL_CHOICES[SELECTED_MODEL_ALIAS]["metric"]

# For debugging/logging
MODEL_RANK = MODEL_CHOICES[SELECTED_MODEL_ALIAS]["rank"]# 

# Optional debug print
print(f"Model Loaded: {EMBEDDING_MODEL_NAME} | dim: {EMBEDDING_DIMENSION} | metric: {SIMILARITY_METRIC}")

# MODEL_CHOICES = {
#     "miniLM": "sentence-transformers/all-MiniLM-L6-v2",
#     "all_mpnet": "sentence-transformers/all-mpnet-base-v2",
#     "multiqa_mpnet": "sentence-transformers/multi-qa-mpnet-base-dot-v1"
# }

# # Use environment variable if set; fallback to default
# import os
# # SELECTED_MODEL_ALIAS = os.getenv("EMBED_MODEL_ALIAS", "miniLM")
# SELECTED_MODEL_ALIAS = os.getenv("EMBED_MODEL_ALIAS", "all_mpnet")
# # SELECTED_MODEL_ALIAS = os.getenv("EMBED_MODEL_ALIAS", "multiqa_mpnet")

# EMBEDDING_MODEL_NAME = MODEL_CHOICES.get(SELECTED_MODEL_ALIAS, MODEL_CHOICES["miniLM"])
# EMBEDDING_DIMENSIONS = {
#     "miniLM": 384,
#     "all_mpnet": 768,
#     "multiqa_mpnet": 768
# }[SELECTED_MODEL_ALIAS]

"""
Configuration module for Business Entity Resolution.
Defines project paths, model hyperparameters, and execution settings.
"""

from pathlib import Path

# Paths
CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# File names
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

CANDIDATE_OUTPUT_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_OUTPUT_PATH = OUTPUT_DIR / "matching_results.tsv"

# Random Seed
RANDOM_SEED = 42

# Embedding Model (MIT Licensed, General Purpose Language Model <= 8B)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Blocking Hyperparameters
BLOCKING_TOP_K_TFIDF = 15
BLOCKING_TOP_K_EMB = 10
TFIDF_NGRAM_RANGE = (3, 5)

# LightGBM Classifier Hyperparameters
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "random_state": RANDOM_SEED,
}
NUM_BOOST_ROUND = 600
EARLY_STOPPING_ROUNDS = 35
N_SPLITS_CV = 5

# Default F0.5 Threshold Range for Tuning
THRESHOLD_GRID = [round(t, 2) for t in [i * 0.01 for i in range(30, 96)]]
DEFAULT_THRESHOLD = 0.65

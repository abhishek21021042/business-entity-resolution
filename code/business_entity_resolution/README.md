# Business Entity Resolution Pipeline

End-to-end entity resolution pipeline for enterprise records across heterogeneous sources (Source 1, Source 2, Source 3).

## Prerequisites & Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Core dependencies:
- `pandas`
- `numpy`
- `scikit-learn`
- `rapidfuzz`
- `scipy`
- `tqdm`

## Project Structure

```
├── dataset/
│   ├── train/
│   └── test/
├── output/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
├── models/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── config.py
│       │   ├── data_loading.py
│       │   ├── normalization.py
│       │   ├── phonetic.py
│       │   ├── blocking.py
│       │   ├── features.py
│       │   ├── train_model.py
│       │   ├── postprocess.py
│       │   ├── evaluate.py
│       │   ├── infer.py
│       │   └── generate_outputs.py
│       ├── utils/
│       │   └── validate_submission.py
│       └── requirements.txt
```

## Running the Pipeline

### 1. Exploratory Data Analysis (EDA)
```bash
python run_eda.py
```

### 2. Verify Local Metric Harness
```bash
python src/evaluate.py
```

### 3. Validate Submission
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

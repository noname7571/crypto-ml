# crypto-ml

End-to-end cryptocurrency ML pipeline — from raw OHLCV data through feature engineering, model training, evaluation and FastAPI deployment, packaged in Docker.

```
Raw OHLCV → Cleaning → Feature Engineering → Model Training → Evaluation → FastAPI → Docker
```

## Stack

| Layer | Technology |
|---|---|
| Data | Binance REST API, Pandas, PyArrow |
| Features | `ta` (RSI, MACD, Bollinger, ATR, OBV, EMA), lag/return/temporal features |
| Models | LSTM (PyTorch) · XGBoost |
| Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Packaging | Docker + docker-compose |
| Tests | pytest |

---

## Project Structure

```
crypto-ml/
├── src/
│   ├── data/
│   │   ├── fetch.py        # Binance OHLCV download & synthetic data helper
│   │   ├── clean.py        # NaN removal, OHLC sanity, dedup, gap-fill
│   │   └── features.py     # Technical indicators + lag/temporal features
│   ├── models/
│   │   ├── lstm.py         # PyTorch LSTM + sequence builder + trainer
│   │   ├── xgboost_model.py# scikit-learn–compatible XGBoost wrapper
│   │   └── evaluate.py     # RMSE, MAE, R², MAPE, directional accuracy
│   ├── training/
│   │   └── train.py        # Full pipeline with MLflow experiment tracking
│   └── api/
│       └── app.py          # FastAPI prediction server
├── tests/                  # pytest test suite
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, XGBoost may require OpenMP runtime:

```bash
brew install libomp
```

### 2. Fetch historical data (optional)

Set your Binance API credentials (public endpoints work without a key):

```bash
export BINANCE_API_KEY=""
export BINANCE_API_SECRET=""
```

```python
from src.data.fetch import fetch_ohlcv, save_ohlcv
from pathlib import Path

df = fetch_ohlcv("BTCUSDT", interval="1h", start="1 Jan, 2022")
save_ohlcv(df, Path("data/raw/BTCUSDT_1h.parquet"))
```

### 3. Train a model

```bash
# XGBoost (default) — uses synthetic data if no parquet file found
python -m src.training.train --model xgboost --data-path data/raw/BTCUSDT_1h.parquet

# LSTM
python -m src.training.train --model lstm --seq-len 24 --data-path data/raw/BTCUSDT_1h.parquet
```

Training writes a serving bundle to `artifacts/model_bundle.pkl`.

### 4. View MLflow experiments

```bash
mlflow ui --backend-store-uri mlruns
# → http://localhost:5000
```

### 5. Serve the API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API auto-loads `artifacts/model_bundle.pkl` if present.
To load a custom bundle path instead:

```bash
MODEL_PATH=/absolute/path/to/model_bundle.pkl uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/ for a simple HTML service page.
For JSON status use http://localhost:8000/status.
For feature preflight use http://localhost:8000/feature-spec.

API docs at **http://localhost:8000/docs**

Example prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [29500.0, 29600.0, ...]}'
```

If your model has `F` features, requests to `/predict` and `/predict/batch`
must send exactly `F` values per instance; otherwise API returns HTTP 422.
The homepage includes a small prediction form that reads `/feature-spec`.

Error responses use a stable shape:

```json
{
  "error": {
    "code": "INVALID_FEATURE_COUNT",
    "message": "Expected 55 features, got 3.",
    "details": null
  }
}
```

### 5b. One-command train + serve

```bash
chmod +x scripts/train_and_serve.sh
./scripts/train_and_serve.sh
```

Optional env vars:

```bash
MODEL=lstm DATA_PATH=data/raw/BTCUSDT_1h.parquet HOST=0.0.0.0 PORT=8000 ./scripts/train_and_serve.sh
```

### 6. Run with Docker

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Prediction API | http://localhost:8000 |
| MLflow UI | http://localhost:5000 |

---

## Running Tests

```bash
pytest
```

---

## Pipeline Overview

### Data Cleaning (`src/data/clean.py`)
- Drops rows with NaN OHLCV values
- Deduplicates by timestamp
- Enforces OHLC sanity (`low ≤ open, close ≤ high`)
- Removes non-positive prices/volumes
- Optional gap-filling with forward-fill

### Feature Engineering (`src/data/features.py`)
- **Returns**: log return, percentage return
- **Rolling**: moving average & std (5/10/20/50), z-score, volume ratio
- **Technical indicators**: RSI-14, MACD, Bollinger Bands, ATR-14, OBV, EMA-20/50
- **Lag features**: close, volume, log return (configurable lag depth)
- **Temporal**: hour-of-day, day-of-week, month (sin/cos encoded)
- **Targets**: `target` (next close, regression) · `target_dir` (up/down, classification)

### Models
| | LSTM | XGBoost |
|---|---|---|
| Input | Sequences `(seq_len × F)` | Flat feature vectors `(F,)` |
| Architecture | 2-layer stacked LSTM + dropout + linear | Gradient-boosted trees (hist method) |
| Training | Adam + ReduceLROnPlateau, gradient clipping | Early stopping on validation RMSE |

### Evaluation Metrics
- RMSE, MAE, R², MAPE
- Directional accuracy (% of time the model correctly calls up/down)

### MLflow Tracking
Every training run logs:
- Hyperparameters
- Per-epoch train/val loss (LSTM)
- Test-set metrics (RMSE, MAE, R², MAPE, directional accuracy)
- Serialised model artifact
"""Stock direction / next-return model (sklearn) — separate from LSTM commodity predictor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.ml.preprocess import preprocess_features
from app.ml.features import build_feature_matrix


TARGET = "target_next_logret"
DROP_FOR_X = {TARGET}


@dataclass
class TrainResult:
    metrics: Dict[str, float]
    feature_importance: List[Dict[str, Any]]
    n_train: int
    n_test: int


MIN_TRAIN_ROWS = 50


def _feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in DROP_FOR_X and not c.startswith("target_")]
    good = [c for c in cols if df[c].notna().sum() >= MIN_TRAIN_ROWS]
    return good if len(good) >= 6 else cols


def train_price_model(feat_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[HistGradientBoostingRegressor, TrainResult]:
    d = preprocess_features(feat_df.copy()).dropna(subset=[TARGET])
    cols = _feature_columns(d)
    d = d.dropna(subset=cols + [TARGET])
    if len(d) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Not enough rows after preprocessing; need at least ~{MIN_TRAIN_ROWS} trading days (got {len(d)})."
        )

    X = d[cols].values
    y = d[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    model = HistGradientBoostingRegressor(
        max_depth=5,
        learning_rate=0.06,
        max_iter=200,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.12,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
    }
    imp = getattr(model, "feature_importances_", None)
    fi = [{"feature": cols[i], "importance": float(imp[i])} for i in range(len(cols))] if imp is not None else []
    fi.sort(key=lambda x: x["importance"], reverse=True)
    tr = TrainResult(metrics=metrics, feature_importance=fi[:20], n_train=len(X_train), n_test=len(X_test))
    return model, tr


def predict_next_logret(model: HistGradientBoostingRegressor, feat_row: pd.Series, columns: List[str]) -> float:
    x = feat_row[columns].values.reshape(1, -1)
    return float(model.predict(x)[0])


def run_price_pipeline(ohlcv: pd.DataFrame) -> Dict[str, Any]:
    """End-to-end: feature engineering → train → next-day prediction on last bar."""
    feat = build_feature_matrix(ohlcv)
    feat = preprocess_features(feat)
    cols = _feature_columns(feat)
    trainable = feat.dropna(subset=cols + [TARGET])
    if len(trainable) < MIN_TRAIN_ROWS:
        return {
            "ok": False,
            "error": f"Insufficient history for model (need ~{MIN_TRAIN_ROWS}+ rows).",
            "rows": len(trainable),
        }

    model, tr = train_price_model(trainable)
    infer_df = feat.dropna(subset=cols)
    last = infer_df.iloc[-1]
    nxt = predict_next_logret(model, last, cols)
    last_close = float(last["close"])
    implied_next = last_close * float(np.exp(nxt))

    return {
        "ok": True,
        "model": "HistGradientBoostingRegressor",
        "task": "regression_next_day_log_return",
        "metrics": tr.metrics,
        "n_train": tr.n_train,
        "n_test": tr.n_test,
        "top_features": tr.feature_importance[:12],
        "last_bar_date": str(feat.index[-1].date()) if hasattr(feat.index[-1], "date") else str(feat.index[-1]),
        "last_close": last_close,
        "predicted_next_log_return": nxt,
        "predicted_next_close_est": round(implied_next, 4),
    }

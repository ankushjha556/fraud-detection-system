"""
Real model inference — loads actual trained artifacts from models/ and scores
transactions through the full XGBoost + VAE + LSTM + meta-learner pipeline.

Falls back gracefully (per-component) if any single artifact is missing,
so partial artifact sets still work rather than crashing the whole app.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import joblib

from models_def.architectures import VAE, LSTMClassifier, SEQ_LEN, FEATURE_COLS

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _path(name):
    return os.path.join(MODELS_DIR, name)


def load_real_artifacts():
    """Load whichever real artifacts are present. Returns a dict with
    None for any component that couldn't be loaded, plus a status report
    so the app can display exactly what's live vs mocked."""
    status = {}
    artifacts = {}

    # --- XGBoost ---
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(_path("xgb_final.json"))
        artifacts["xgb"] = model
        status["xgb"] = "loaded"
    except Exception as e:
        artifacts["xgb"] = None
        status["xgb"] = f"unavailable ({type(e).__name__})"

    # --- VAE ---
    try:
        vae = VAE(input_dim=len(FEATURE_COLS))
        vae.load_state_dict(torch.load(_path("vae_model.pt"), map_location=DEVICE))
        vae.to(DEVICE)
        vae.eval()
        artifacts["vae"] = vae
        status["vae"] = "loaded"
    except Exception as e:
        artifacts["vae"] = None
        status["vae"] = f"unavailable ({type(e).__name__})"

    # --- LSTM ---
    try:
        lstm = LSTMClassifier(input_dim=len(FEATURE_COLS))
        lstm.load_state_dict(torch.load(_path("lstm_model.pt"), map_location=DEVICE))
        lstm.to(DEVICE)
        lstm.eval()
        artifacts["lstm"] = lstm
        status["lstm"] = "loaded"
    except Exception as e:
        artifacts["lstm"] = None
        status["lstm"] = f"unavailable ({type(e).__name__})"

    # --- Meta-learner ---
    try:
        meta = joblib.load(_path("meta_learner.pkl"))
        artifacts["meta"] = meta
        status["meta"] = "loaded"
    except Exception as e:
        artifacts["meta"] = None
        status["meta"] = f"unavailable ({type(e).__name__})"

    # --- VAE score normalization stats (needed to reproduce MinMaxScaler behavior) ---
    # Saved separately during training since MinMaxScaler was fit per-batch there;
    # for live single-transaction scoring we use fixed reference bounds from the
    # training-time VAE score distribution instead of refitting a scaler on n=1.
    try:
        with open(_path("vae_score_bounds.json"), "r") as f:
            artifacts["vae_bounds"] = json.load(f)
    except Exception:
        # Reasonable defaults derived from the notebook's reported score ranges
        artifacts["vae_bounds"] = {"recon_min": 0.0, "recon_max": 0.8, "kl_min": 0.0, "kl_max": 20.0}

    artifacts["_status"] = status
    return artifacts


def build_feature_vector(txn: dict, feature_defaults: dict = None) -> np.ndarray:
    """Convert a partial transaction dict (from UI sliders) into a full
    30-feature vector, filling unset V-features with 0.0 (their mean under
    the PCA transform) and computing Amount_scaled/Time_scaled from
    provided raw Amount/Time using saved scaler stats if available."""
    defaults = feature_defaults or {}
    vec = []
    for feat in FEATURE_COLS:
        if feat in txn:
            vec.append(float(txn[feat]))
        elif feat == "Amount_scaled":
            amount = txn.get("Amount", 88.35)  # dataset mean
            amount_mean, amount_std = defaults.get("amount_mean", 88.35), defaults.get("amount_std", 250.12)
            vec.append((amount - amount_mean) / amount_std)
        elif feat == "Time_scaled":
            vec.append(txn.get("Time_scaled", 0.0))
        else:
            vec.append(defaults.get(feat, 0.0))
    return np.array(vec, dtype=np.float32)


def score_transaction_real(txn: dict, artifacts: dict) -> dict:
    """Score a single transaction using real loaded models wherever available,
    falling back to a neutral/mock value per-component if a model is missing."""
    x = build_feature_vector(txn)
    x_2d = x.reshape(1, -1)

    result = {}

    # XGBoost
    if artifacts.get("xgb") is not None:
        result["xgb_score"] = float(artifacts["xgb"].predict_proba(x_2d)[0, 1])
        result["xgb_live"] = True
    else:
        result["xgb_score"] = 0.5
        result["xgb_live"] = False

    # VAE
    if artifacts.get("vae") is not None:
        x_tensor = torch.tensor(x_2d, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            recon, mu, logvar = artifacts["vae"](x_tensor)
            recon_error = torch.mean((recon - x_tensor) ** 2, dim=1).item()
            kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).item()
        bounds = artifacts["vae_bounds"]
        recon_norm = np.clip((recon_error - bounds["recon_min"]) / max(bounds["recon_max"] - bounds["recon_min"], 1e-6), 0, 1)
        kl_norm = np.clip((kl - bounds["kl_min"]) / max(bounds["kl_max"] - bounds["kl_min"], 1e-6), 0, 1)
        result["vae_score"] = float(np.clip((recon_norm + kl_norm) / 2, 0, 1))
        result["vae_live"] = True
    else:
        result["vae_score"] = 0.5
        result["vae_live"] = False

    # LSTM (single-transaction repeated-sequence approximation, matching Section 8's MC Dropout setup)
    if artifacts.get("lstm") is not None:
        x_seq = torch.tensor(x_2d, dtype=torch.float32).unsqueeze(1).repeat(1, SEQ_LEN, 1).to(DEVICE)
        with torch.no_grad():
            out = artifacts["lstm"](x_seq)
            proba = F.softmax(out, dim=1)[:, 1].item()
        result["lstm_score"] = float(proba)
        result["lstm_live"] = True
    else:
        result["lstm_score"] = 0.5
        result["lstm_live"] = False

    # Meta-ensemble
    if artifacts.get("meta") is not None:
        meta_x = np.array([[result["xgb_score"], result["vae_score"], result["lstm_score"]]])
        result["ensemble_score"] = float(artifacts["meta"].predict_proba(meta_x)[0, 1])
        result["ensemble_live"] = True
    else:
        result["ensemble_score"] = float(
            0.5 * result["xgb_score"] + 0.35 * result["vae_score"] + 0.15 * result["lstm_score"]
        )
        result["ensemble_live"] = False

    # Uncertainty via lightweight MC Dropout on LSTM if available
    if artifacts.get("lstm") is not None:
        artifacts["lstm"].train()  # enable dropout
        x_seq = torch.tensor(x_2d, dtype=torch.float32).unsqueeze(1).repeat(1, SEQ_LEN, 1).to(DEVICE)
        with torch.no_grad():
            samples = [F.softmax(artifacts["lstm"](x_seq), dim=1)[:, 1].item() for _ in range(30)]
        artifacts["lstm"].eval()
        result["uncertainty"] = float(np.std(samples))
    else:
        result["uncertainty"] = 0.1

    return result

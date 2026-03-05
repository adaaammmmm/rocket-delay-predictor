# 🚀 Rocket Launch Delay Predictor

A complete end-to-end machine learning project predicting whether a rocket launch will be delayed — and by how long — using a two-stage XGBoost + LightGBM pipeline with full SHAP explainability.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📁 Project Structure

```
rocket_delay_predictor/
├── data/
│   ├── raw/                   # Raw dataset (generated)
│   ├── processed/             # Processed splits
│   └── generate_data.py       # Synthetic data generator
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   └── 02_model_analysis.ipynb# Model evaluation & SHAP
├── src/
│   ├── data_pipeline.py       # Feature engineering + sklearn pipelines
│   ├── train.py               # Model training (XGBoost + Optuna + LightGBM)
│   └── explain.py             # SHAP explainability plots
├── app/
│   └── streamlit_app.py       # Interactive dashboard
├── models/                    # Saved model artefacts (after training)
├── outputs/                   # Generated plots
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Clone & install
git clone https://github.com/yourname/rocket-delay-predictor
cd rocket_delay_predictor
pip install -r requirements.txt

# 2. Generate synthetic dataset (~4,600 launches)
python data/generate_data.py

# 3. Train all models (~2–3 min)
python src/train.py

# 4. Generate SHAP explanation plots
python src/explain.py

# 5. Launch the interactive dashboard
streamlit run app/streamlit_app.py
```

---

## 🏗️ Model Architecture

### Stage 1 — Delay Classifier (XGBoost)
- **Goal:** Predict *whether* a launch will be delayed (binary)
- **Handling imbalance:** `scale_pos_weight` tuned automatically
- **Tuning:** Optuna with 50 trials, StratifiedKFold(5) cross-validation
- **Metric:** ROC-AUC

### Stage 2 — Delay Duration Regressor (LightGBM)
- **Goal:** Predict *how many hours* the delay will last
- **Only applied** when Stage 1 predicts a delay (probability > 30%)
- **Metrics:** MAE, RMSE (hours)

---

## 🔧 Features Used

| Category | Feature | Description |
|---|---|---|
| Historical | `provider_success_rate` | Rolling success % for the company |
| Historical | `site_success_rate` | Rolling success % for the launch site |
| Rocket | `rocket_age_years` | Years since first flight |
| Weather | `wind_speed_kmh` | Wind speed at launch site on launch day |
| Weather | `temp_celsius` | Temperature |
| Weather | `precipitation_mm` | Rainfall |
| Weather | `cloud_cover_pct` | Cloud coverage |
| Mission | `mission_type` | Satellite / ISS / Crewed / etc. |
| Mission | `is_crewed` | Boolean — extra scrutiny for crewed flights |
| Context | `launches_this_month` | Simultaneous launch congestion |
| Derived | `weather_severity` | Composite weather risk score |
| Derived | `season` | Season based on launch month |
| Derived | `high_wind` | Boolean flag (>40 km/h) |

---

## 📊 Key Results

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | ~0.72 | ~0.40 | ~0.45 | ~0.35 |
| Decision Tree | ~0.68 | ~0.42 | ~0.40 | ~0.44 |
| **XGBoost (Tuned)** | **~0.87** | **~0.65** | **~0.70** | **~0.60** |

Regressor MAE: ~8–12 hours

---

## 🔍 SHAP Findings

Based on SHAP analysis, the most impactful features are:

1. **`provider_success_rate`** — Strongest predictor. Low historical success → high delay risk
2. **`site_success_rate`** — Site track record matters nearly as much
3. **`wind_speed_kmh`** — Delay risk increases sharply above ~40 km/h
4. **`rocket_age_years`** — Older rockets carry marginally higher risk
5. **`precipitation_mm`** — Even light rain contributes measurably

> *"Provider success history is the single strongest predictor of delays, but wind speed above 40 km/h becomes the decisive factor in marginal cases."*

---

## 🖥️ Dashboard Features

The Streamlit app provides:
- **Predict Tab:** Input all launch conditions → get delay probability gauge + expected delay hours + SHAP waterfall chart
- **Data Explorer Tab:** Interactive EDA charts (launches over time, delay by company, weather distributions)
- **About Tab:** Architecture documentation

---

## 🚀 Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set main file as `app/streamlit_app.py`
4. Deploy (free tier)

---

## 📚 Tech Stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `lightgbm` · `shap` · `optuna` · `imbalanced-learn` · `streamlit` · `plotly` · `matplotlib` · `seaborn`

---

## 📄 License

MIT — free to use, modify, and distribute.

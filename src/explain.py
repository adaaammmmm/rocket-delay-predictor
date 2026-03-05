"""
src/explain.py
Generates SHAP explanations and saves plots to outputs/.

Run:  python src/explain.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.data_pipeline import load_and_split, get_feature_names

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

PALETTE = {
    "bg":      "#0d1117",
    "surface": "#161b22",
    "accent":  "#58a6ff",
    "danger":  "#f85149",
    "success": "#3fb950",
    "text":    "#e6edf3",
    "muted":   "#8b949e",
}


def style_fig(fig, ax_list=None):
    fig.patch.set_facecolor(PALETTE["bg"])
    if ax_list:
        for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
            ax.set_facecolor(PALETTE["surface"])
            ax.tick_params(colors=PALETTE["text"])
            ax.xaxis.label.set_color(PALETTE["text"])
            ax.yaxis.label.set_color(PALETTE["text"])
            ax.title.set_color(PALETTE["text"])
            for spine in ax.spines.values():
                spine.set_edgecolor(PALETTE["muted"])


def load_models():
    clf   = joblib.load(os.path.join(MODELS_DIR, "xgb_classifier.pkl"))
    prep  = joblib.load(os.path.join(MODELS_DIR, "clf_preprocessor.pkl"))
    fnames= joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    return clf, prep, fnames


# ── Plot 1: SHAP Summary (Beeswarm) ──────────────────────────────────────────

def plot_summary(shap_values, X_transformed, feature_names, top_n=20):
    print("  Generating SHAP summary plot…")
    fig, ax = plt.subplots(figsize=(10, 8))
    style_fig(fig, ax)

    top_idx = np.argsort(np.abs(shap_values).mean(axis=0))[-top_n:][::-1]
    sv_top  = shap_values[:, top_idx]
    fn_top  = [feature_names[i] for i in top_idx]

    # Beeswarm-style horizontal scatter
    for row_i, (sv_col, fname) in enumerate(zip(sv_top.T, fn_top)):
        jitter = np.random.normal(0, 0.08, len(sv_col))
        colors = [PALETTE["danger"] if v > 0 else PALETTE["success"] for v in sv_col]
        ax.scatter(sv_col, row_i + jitter, c=colors, alpha=0.4, s=6, zorder=2)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(fn_top, fontsize=9, color=PALETTE["text"])
    ax.axvline(0, color=PALETTE["muted"], linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value  (impact on delay probability)", color=PALETTE["text"])
    ax.set_title("Feature Importance — SHAP Summary", color=PALETTE["text"], fontsize=13, pad=12)
    ax.grid(axis="x", alpha=0.15, color=PALETTE["muted"])

    red_patch   = mpatches.Patch(color=PALETTE["danger"],  label="Increases delay risk")
    green_patch = mpatches.Patch(color=PALETTE["success"], label="Decreases delay risk")
    ax.legend(handles=[red_patch, green_patch], facecolor=PALETTE["surface"],
              labelcolor=PALETTE["text"], fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved → {path}")
    return path


# ── Plot 2: SHAP Dependence — Wind Speed ─────────────────────────────────────

def plot_dependence(shap_values, X_transformed, feature_names):
    print("  Generating SHAP dependence plot (wind speed)…")

    try:
        wi = feature_names.index("wind_speed_kmh")
    except ValueError:
        print("  wind_speed_kmh not found, skipping.")
        return None

    wind_vals = X_transformed[:, wi]
    wind_shap = shap_values[:, wi]

    fig, ax = plt.subplots(figsize=(9, 5))
    style_fig(fig, ax)

    sc = ax.scatter(wind_vals, wind_shap, c=wind_shap, cmap="RdYlGn_r",
                    s=8, alpha=0.5, vmin=-0.3, vmax=0.3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("SHAP value", color=PALETTE["text"])
    cbar.ax.yaxis.set_tick_params(color=PALETTE["text"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["text"])

    ax.axhline(0, color=PALETTE["muted"], linewidth=0.8, linestyle="--")
    ax.axvline(40, color=PALETTE["danger"], linewidth=1.2, linestyle=":",
               label="High-wind threshold (40 km/h)")
    ax.set_xlabel("Wind Speed (km/h)")
    ax.set_ylabel("SHAP value")
    ax.set_title("Wind Speed vs Delay Risk (SHAP Dependence)", fontsize=12)
    ax.legend(facecolor=PALETTE["surface"], labelcolor=PALETTE["text"], fontsize=9)
    ax.grid(alpha=0.12, color=PALETTE["muted"])

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "shap_dependence_wind.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved → {path}")
    return path


# ── Plot 3: Waterfall for single prediction ───────────────────────────────────

def plot_waterfall(explainer, shap_values, X_transformed, feature_names,
                   sample_idx=None, label="high_risk"):
    print(f"  Generating waterfall plot (sample {sample_idx})…")

    if sample_idx is None:
        # Pick the sample with highest delay probability
        sample_idx = int(np.argmax(shap_values.sum(axis=1)))

    sv    = shap_values[sample_idx]
    top_n = 12
    top_i = np.argsort(np.abs(sv))[-top_n:][::-1]
    top_sv = sv[top_i]
    top_fn = [feature_names[i] for i in top_i]

    fig, ax = plt.subplots(figsize=(10, 6))
    style_fig(fig, ax)

    colors = [PALETTE["danger"] if v > 0 else PALETTE["success"] for v in top_sv]
    bars = ax.barh(range(top_n), top_sv[::-1], color=colors[::-1], height=0.6, zorder=2)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_fn[::-1], fontsize=9, color=PALETTE["text"])
    ax.axvline(0, color=PALETTE["muted"], linewidth=0.8)
    ax.set_xlabel("SHAP value (contribution to delay probability)")
    ax.set_title(f"Why was this launch flagged? — Waterfall Explanation", fontsize=12)
    ax.grid(axis="x", alpha=0.13, color=PALETTE["muted"])

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, f"shap_waterfall_{label}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved → {path}")
    return path


# ── Plot 4: Model comparison bar chart ───────────────────────────────────────

def plot_model_comparison():
    import json
    summary_path = os.path.join(MODELS_DIR, "training_summary.json")
    if not os.path.exists(summary_path):
        print("  No training_summary.json found, skipping comparison plot.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    results = summary["classifier_results"]
    models  = [r["model"] for r in results]
    metrics = ["accuracy", "roc_auc", "f1", "precision", "recall"]
    colors  = [PALETTE["accent"], PALETTE["success"], PALETTE["danger"], "#f0a500", "#c561f6"]

    x     = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(11, 5))
    style_fig(fig, ax)

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [r[metric] for r in results]
        ax.bar(x + i * width, vals, width, label=metric.upper(), color=color, alpha=0.85)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, fontsize=10, color=PALETTE["text"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Classification Metrics", fontsize=12)
    ax.legend(facecolor=PALETTE["surface"], labelcolor=PALETTE["text"], fontsize=9,
              loc="upper right")
    ax.grid(axis="y", alpha=0.13, color=PALETTE["muted"])
    ax.axhline(1.0, color=PALETTE["muted"], linewidth=0.5, linestyle="--")

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def compute_shap(X_test_t, clf, feature_names, n_samples=500):
    print("  Computing SHAP values (TreeExplainer)…")
    explainer   = shap.TreeExplainer(clf)
    X_sample    = X_test_t[:n_samples]
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class
    return explainer, shap_values, X_sample


def main():
    print("\n🔍 Generating SHAP Explainability Plots\n")

    clf, prep, feature_names = load_models()
    splits = load_and_split()
    X_train, X_test, y_train, y_test = splits["clf"]

    X_test_t = prep.transform(X_test)
    explainer, shap_values, X_sample = compute_shap(X_test_t, clf, feature_names)

    plot_summary(shap_values, X_sample, feature_names)
    plot_dependence(shap_values, X_sample, feature_names)
    plot_waterfall(explainer, shap_values, X_sample, feature_names, label="highest_risk")

    # Waterfall for a low-risk launch
    low_risk_idx = int(np.argmin(shap_values.sum(axis=1)))
    plot_waterfall(explainer, shap_values, X_sample, feature_names,
                   sample_idx=low_risk_idx, label="lowest_risk")

    plot_model_comparison()

    print(f"\n✅ All plots saved to {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()

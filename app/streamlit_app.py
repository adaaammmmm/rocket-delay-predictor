"""
app/streamlit_app.py
Rocket Launch Delay Predictor — Interactive Dashboard

Run:  streamlit run app/streamlit_app.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚀 Rocket Launch Delay Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0d1117; }
  .block-container { padding-top: 1.5rem; }
  h1, h2, h3 { color: #58a6ff; }
  .metric-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-val  { font-size: 2.2rem; font-weight: 700; }
  .metric-label{ font-size: 0.85rem; color: #8b949e; margin-top: 4px; }
  .risk-high   { color: #f85149; }
  .risk-medium { color: #f0a500; }
  .risk-low    { color: #3fb950; }
  .stSlider > div { color: #e6edf3; }
  div[data-testid="stSidebar"] { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
DATA_PATH   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "raw", "launches.csv")

COMPANIES = ["SpaceX", "Roscosmos", "ULA", "Arianespace", "ISRO",
             "JAXA", "Blue Origin", "Rocket Lab", "Northrop Grumman", "Virgin Orbit"]
ROCKETS   = ["Falcon 9", "Falcon Heavy", "Soyuz", "Atlas V", "Vulcan",
             "Ariane 5", "Ariane 6", "PSLV", "H-IIA", "New Shepard",
             "Electron", "Antares", "LauncherOne"]
SITES     = ["Cape Canaveral, FL", "Vandenberg, CA", "Baikonur, Kazakhstan",
             "Kourou, French Guiana", "Satish Dhawan, India", "Tanegashima, Japan",
             "Wallops Island, VA", "Mahia, New Zealand"]
MISSIONS  = ["Satellite", "ISS Resupply", "Science", "Military", "Commercial", "Crewed", "Test"]

ROCKET_FIRST_FLIGHTS = {
    "Falcon 9": 2010, "Falcon Heavy": 2018, "Soyuz": 1966, "Atlas V": 2002,
    "Vulcan": 2024, "Ariane 5": 1996, "Ariane 6": 2024, "PSLV": 1993,
    "H-IIA": 2001, "New Shepard": 2015, "Electron": 2017, "Antares": 2013,
    "LauncherOne": 2020,
}


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        clf       = joblib.load(os.path.join(MODELS_DIR, "xgb_classifier.pkl"))
        prep_clf  = joblib.load(os.path.join(MODELS_DIR, "clf_preprocessor.pkl"))
        lgb       = joblib.load(os.path.join(MODELS_DIR, "lgb_regressor.pkl"))
        prep_reg  = joblib.load(os.path.join(MODELS_DIR, "reg_preprocessor.pkl"))
        fnames    = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
        return clf, prep_clf, lgb, prep_reg, fnames, True
    except FileNotFoundError:
        return None, None, None, None, None, False


@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=["launch_date"])
        return df
    except FileNotFoundError:
        return None


# ── Feature builder ───────────────────────────────────────────────────────────
def build_input_df(company, rocket, site, mission_type, launch_year, launch_month,
                   wind_speed, temp, precipitation, cloud_cover, is_crewed,
                   prov_rate, site_rate, launches_month):
    age = launch_year - ROCKET_FIRST_FLIGHTS.get(rocket, 2010)
    return pd.DataFrame([{
        "company":               company,
        "rocket":                rocket,
        "site":                  site,
        "mission_type":          mission_type,
        "launch_month":          launch_month,
        "rocket_age_years":      age,
        "provider_success_rate": prov_rate,
        "site_success_rate":     site_rate,
        "launches_this_month":   launches_month,
        "temp_celsius":          temp,
        "wind_speed_kmh":        wind_speed,
        "precipitation_mm":      precipitation,
        "cloud_cover_pct":       cloud_cover,
        "is_crewed":             int(is_crewed),
        "season":                1 if launch_month in [12,1,2] else (2 if launch_month in [3,4,5] else (3 if launch_month in [6,7,8] else 4)),
        "high_wind":             int(wind_speed > 40),
        "weather_severity":      wind_speed/100 + precipitation/20 + cloud_cover/200,
        "decade":                str(launch_year // 10 * 10),
    }])


# ── Gauge chart ───────────────────────────────────────────────────────────────
def delay_gauge(prob: float) -> go.Figure:
    if prob < 0.25:
        bar_color, label = "#3fb950", "LOW RISK"
    elif prob < 0.55:
        bar_color, label = "#f0a500", "MEDIUM RISK"
    else:
        bar_color, label = "#f85149", "HIGH RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": bar_color}},
        delta={"reference": 15, "valueformat": ".1f"},
        title={"text": f"Delay Probability<br><span style='font-size:1.1rem;color:{bar_color}'>{label}</span>",
               "font": {"size": 16, "color": "#e6edf3"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b949e",
                     "tickfont": {"color": "#8b949e"}},
            "bar":  {"color": bar_color, "thickness": 0.3},
            "bgcolor": "#161b22",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  25], "color": "#0d2818"},
                {"range": [25, 55], "color": "#1f1900"},
                {"range": [55,100], "color": "#200a0a"},
            ],
            "threshold": {
                "line":  {"color": "#e6edf3", "width": 2},
                "thickness": 0.8,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(
        height=280,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        margin=dict(l=30, r=30, t=50, b=10),
    )
    return fig


# ── SHAP waterfall (fast inline) ──────────────────────────────────────────────
def shap_waterfall_fig(shap_vals, feature_names, top_n=12):
    idx    = np.argsort(np.abs(shap_vals))[-top_n:][::-1]
    sv     = shap_vals[idx]
    fn     = [feature_names[i] for i in idx]

    colors = ["#f85149" if v > 0 else "#3fb950" for v in sv]
    fig = go.Figure(go.Bar(
        x=sv[::-1], y=fn[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=[f"{v:+.3f}" for v in sv[::-1]],
        textposition="outside",
        textfont={"color": "#e6edf3", "size": 11},
    ))
    fig.update_layout(
        title="Feature Contributions (SHAP Waterfall)",
        title_font={"color": "#58a6ff", "size": 14},
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        height=420,
        margin=dict(l=10, r=80, t=50, b=10),
        xaxis=dict(
            title="SHAP value",
            titlefont={"color": "#e6edf3"},
            tickfont={"color": "#8b949e"},
            gridcolor="#30363d",
            zeroline=True, zerolinecolor="#8b949e",
        ),
        yaxis=dict(tickfont={"color": "#e6edf3"}),
    )
    return fig


# ── EDA Charts ────────────────────────────────────────────────────────────────
def plot_eda(df: pd.DataFrame):
    col1, col2 = st.columns(2)

    with col1:
        # Launches per year
        yearly = df.groupby("launch_year").size().reset_index(name="count")
        fig = px.bar(yearly, x="launch_year", y="count",
                     title="Launches Per Year",
                     color="count", color_continuous_scale="Blues")
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                          font_color="#e6edf3", coloraxis_showscale=False,
                          xaxis=dict(gridcolor="#30363d"),
                          yaxis=dict(gridcolor="#30363d"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Delay rate by company
        comp = df.groupby("company")["delayed"].mean().reset_index()
        comp.columns = ["company", "delay_rate"]
        comp = comp.sort_values("delay_rate", ascending=False)
        colors = ["#f85149" if v > 0.15 else "#f0a500" if v > 0.08 else "#3fb950"
                  for v in comp["delay_rate"]]
        fig = go.Figure(go.Bar(
            x=comp["company"], y=comp["delay_rate"] * 100,
            marker_color=colors, text=[f"{v:.1%}" for v in comp["delay_rate"]],
            textposition="outside",
        ))
        fig.update_layout(
            title="Delay Rate by Company (%)",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3", yaxis_title="Delay %",
            xaxis=dict(tickangle=-30, gridcolor="#30363d"),
            yaxis=dict(gridcolor="#30363d"),
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        # Wind speed distribution for delayed vs not
        fig = go.Figure()
        for label, color, name in [(0, "#3fb950", "On Time"), (1, "#f85149", "Delayed")]:
            subset = df[df["delayed"] == label]["wind_speed_kmh"]
            fig.add_trace(go.Histogram(
                x=subset, name=name, opacity=0.65,
                marker_color=color, nbinsx=40,
            ))
        fig.update_layout(
            barmode="overlay",
            title="Wind Speed Distribution: Delayed vs On Time",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#e6edf3",
            legend=dict(bgcolor="#161b22"),
            xaxis=dict(title="Wind Speed (km/h)", gridcolor="#30363d"),
            yaxis=dict(gridcolor="#30363d"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Monthly delay rate
        monthly = df.groupby("launch_month")["delayed"].mean().reset_index()
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly["month_name"] = monthly["launch_month"].apply(lambda x: month_names[x-1])
        fig = px.line(monthly, x="month_name", y="delayed",
                      title="Delay Rate by Month",
                      markers=True, color_discrete_sequence=["#58a6ff"])
        fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                          font_color="#e6edf3", yaxis_title="Delay Rate",
                          xaxis=dict(gridcolor="#30363d"),
                          yaxis=dict(gridcolor="#30363d"))
        st.plotly_chart(fig, use_container_width=True)


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    clf, prep_clf, lgb_reg, prep_reg, fnames, models_ready = load_models()
    df = load_data()

    # Header
    st.title("🚀 Rocket Launch Delay Predictor")
    st.markdown(
        "<p style='color:#8b949e'>Predict delay probability for any rocket launch. "
        "Powered by XGBoost + SHAP explainability.</p>",
        unsafe_allow_html=True,
    )

    if not models_ready:
        st.warning("⚠️  Models not found. Run `python src/train.py` first, then refresh.")

    # Tabs
    tab_pred, tab_eda, tab_about = st.tabs(["🎯 Predict", "📊 Data Explorer", "ℹ️ About"])

    # ─── TAB 1: PREDICTOR ────────────────────────────────────────────────────
    with tab_pred:
        st.markdown("### Configure Your Launch")
        c1, c2, c3 = st.columns(3)

        with c1:
            company      = st.selectbox("Launch Provider", COMPANIES)
            rocket       = st.selectbox("Rocket", ROCKETS)
            site         = st.selectbox("Launch Site", SITES)
            mission_type = st.selectbox("Mission Type", MISSIONS)
            is_crewed    = st.checkbox("Crewed Mission", value=False)

        with c2:
            launch_year  = st.slider("Launch Year",    2000, 2030, 2024)
            launch_month = st.slider("Launch Month",      1,   12,    6,
                                     format="%d", help="1=Jan … 12=Dec")
            wind_speed   = st.slider("Wind Speed (km/h)", 0.0, 100.0, 15.0, 0.5)
            temp         = st.slider("Temperature (°C)", -20.0, 45.0, 22.0, 0.5)

        with c3:
            precipitation = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0, 0.5)
            cloud_cover   = st.slider("Cloud Cover (%)",    0.0, 100.0, 30.0, 1.0)
            prov_rate     = st.slider("Provider Hist. Success Rate", 0.0, 1.0, 0.90, 0.01)
            site_rate     = st.slider("Site Hist. Success Rate",     0.0, 1.0, 0.90, 0.01)
            launches_mo   = st.slider("Other Launches This Month",    0,   20,    3)

        st.markdown("---")

        if st.button("🚀 Predict Delay Risk", type="primary", use_container_width=True):
            if not models_ready:
                st.error("Models not loaded. Run training first.")
            else:
                input_df = build_input_df(
                    company, rocket, site, mission_type, launch_year, launch_month,
                    wind_speed, temp, precipitation, cloud_cover, is_crewed,
                    prov_rate, site_rate, launches_mo,
                )

                X_t       = prep_clf.transform(input_df)
                delay_prob= clf.predict_proba(X_t)[0][1]

                X_t_reg   = prep_reg.transform(input_df)
                delay_hrs = lgb_reg.predict(X_t_reg)[0] if delay_prob > 0.3 else 0.0
                delay_hrs = max(0.0, delay_hrs)

                # ── Results row ──
                g1, g2, g3 = st.columns([2, 1, 1])
                with g1:
                    st.plotly_chart(delay_gauge(delay_prob), use_container_width=True)
                with g2:
                    risk_class = "risk-high" if delay_prob > 0.55 else \
                                 "risk-medium" if delay_prob > 0.25 else "risk-low"
                    st.markdown(f"""
                    <div class='metric-box' style='margin-top:50px'>
                        <div class='metric-val {risk_class}'>{delay_prob*100:.1f}%</div>
                        <div class='metric-label'>Delay Probability</div>
                    </div>""", unsafe_allow_html=True)
                with g3:
                    if delay_prob > 0.3:
                        st.markdown(f"""
                        <div class='metric-box' style='margin-top:50px'>
                            <div class='metric-val' style='color:#f0a500'>{delay_hrs:.1f}h</div>
                            <div class='metric-label'>Expected Delay Duration</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class='metric-box' style='margin-top:50px'>
                            <div class='metric-val' style='color:#3fb950'>✓</div>
                            <div class='metric-label'>On-Time Expected</div>
                        </div>""", unsafe_allow_html=True)

                # ── SHAP explanation ──
                st.markdown("#### Why this prediction?")
                try:
                    explainer   = shap.TreeExplainer(clf)
                    shap_values = explainer.shap_values(X_t)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    sv = shap_values[0]
                    st.plotly_chart(shap_waterfall_fig(sv, fnames), use_container_width=True)
                except Exception as e:
                    st.info(f"SHAP computation: {e}")

                # ── Key insights ──
                st.markdown("#### 🔑 Key Risk Factors")
                ic1, ic2, ic3 = st.columns(3)
                with ic1:
                    wc = "🔴" if wind_speed > 40 else "🟡" if wind_speed > 25 else "🟢"
                    st.info(f"{wc} **Wind**: {wind_speed} km/h {'— HIGH RISK' if wind_speed>40 else '— Moderate' if wind_speed>25 else '— Safe'}")
                with ic2:
                    rc = "🔴" if prov_rate < 0.8 else "🟡" if prov_rate < 0.9 else "🟢"
                    st.info(f"{rc} **Provider Record**: {prov_rate:.0%} success rate")
                with ic3:
                    age = launch_year - ROCKET_FIRST_FLIGHTS.get(rocket, 2010)
                    ac = "🟡" if age > 15 else "🟢"
                    st.info(f"{ac} **Rocket Age**: {age} years old")

    # ─── TAB 2: EDA ──────────────────────────────────────────────────────────
    with tab_eda:
        if df is None:
            st.warning("Data not found. Run `python data/generate_data.py` first.")
        else:
            st.markdown("### Launch Dataset Overview")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Total Launches",  f"{len(df):,}")
            mc2.metric("Delay Rate",       f"{df['delayed'].mean():.1%}")
            mc3.metric("Companies",        df["company"].nunique())
            mc4.metric("Launch Sites",     df["site"].nunique())
            st.markdown("---")
            plot_eda(df)
            with st.expander("📋 Raw Data Sample"):
                st.dataframe(df.head(100), use_container_width=True)

    # ─── TAB 3: ABOUT ────────────────────────────────────────────────────────
    with tab_about:
        st.markdown("""
### About This Project

This end-to-end data science project predicts rocket launch delays using machine learning.

#### Architecture
- **Stage 1 — Classifier:** XGBoost trained to predict *whether* a launch will be delayed
  - Handles class imbalance via `scale_pos_weight`
  - Hyperparameter tuning with Optuna (50 trials, StratifiedKFold CV)
- **Stage 2 — Regressor:** LightGBM trained (on delayed launches only) to predict *how long* the delay will be

#### Features Used
| Category | Features |
|---|---|
| Historical | Provider success rate, site success rate |
| Rocket | Age in years |
| Weather | Wind speed, temperature, precipitation, cloud cover |
| Mission | Type (crewed, cargo, etc.), month, year |
| Context | Concurrent launches this month |

#### Explainability
SHAP (SHapley Additive exPlanations) values are computed via `TreeExplainer` 
to show exactly which features drove each individual prediction.

#### Stack
`pandas` · `scikit-learn` · `xgboost` · `lightgbm` · `shap` · `optuna` · `streamlit` · `plotly`

#### How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data
python data/generate_data.py

# 3. Train models (~2-3 minutes)
python src/train.py

# 4. Generate SHAP plots
python src/explain.py

# 5. Launch dashboard
streamlit run app/streamlit_app.py
```
        """)


if __name__ == "__main__":
    main()

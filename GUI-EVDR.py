# ======================================================================
# EVDR GUI (Streamlit) - XGBoost + MLP only
# Keep same inputs/layout as DI GUI, update plot for EVDR
# ======================================================================

import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import base64
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ======================================================================
# PAGE CONFIG + CSS
# ======================================================================
st.set_page_config(page_title="EVDR Prediction", layout="wide", page_icon="")

st.markdown(r"""
<style>
    .block-container { padding-top: 3rem; }

    .stNumberInput > div > div, .stSelectbox > div > div {
        max-width: 240px !important;
    }
    .stNumberInput label, .stSelectbox label {
        font-size: 50px !important;
        font-weight: 1000;
    }

    .section-header {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    .form-banner {
        text-align: center;
        background: linear-gradient(pink);
        padding: 0.0rem;
        font-size: 50px;
        font-weight: 800;
        color: black;
        border-radius: 0px;
        margin: 0rem 0;
    }
    .prediction-result {
        font-size: 25px;
        font-weight: bold;
        color: black;
        background-color: lightgray;
        padding: 0.2rem;
        border-radius: 0px;
        text-align: left;
        margin-top: 0rem;
    }

    /* Buttons (keep your look) */
    div.stButton > button {
        background-color: pink;
        color: black;
        font-weight: bold;
        font-size: 20px;
        border-radius: 5px;
        padding: 0rem 3.0rem;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #27ae60;
    }

    /* ===== MAIN 2-COLUMN ROW ONLY (col1 background) ===== */
    /* If :has() ever fails on an older environment, Streamlit will just ignore it (safe). */
    div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(2))
      > div[data-testid="column"]:first-child
      > div {
        background-color: white;   /* <-- left panel background (change if you want) */
        border-radius: 16px;
        padding: 20px 20px;
        border: 1px solid white;
    }

    /* ===== CANCEL the background for any nested columns (buttons etc.) ===== */
    div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"]
      > div[data-testid="column"] > div {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================================
# HELPERS
# ======================================================================
def normalize_input(x_raw, scaler):
    return scaler.transform(x_raw)

def denormalize_output(y_scaled, scaler):
    y_scaled = np.array(y_scaled).reshape(-1, 1)
    return scaler.inverse_transform(y_scaled)[0][0]


# ======================================================================
# LOAD MODELS (Streamlit safe)
# ======================================================================
@st.cache_resource
def load_models():
    # XGBoost
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model("Best_XGBoost_Model.json")

    # MLP + scalers
    mlp_model = load_model("ANN_MLP_Model.keras")
    mlp_scaler_X = joblib.load("ANN_MLP_Scaler_X.save")
    mlp_scaler_y = joblib.load("ANN_MLP_Scaler_y.save")

    return {
        "XGBoost": xgb_model,
        "MLP": mlp_model,
        "_MLP_scaler_X": mlp_scaler_X,
        "_MLP_scaler_y": mlp_scaler_y,
    }

models = load_models()

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

# ======================================================================
# MAIN TWO-COLUMN LAYOUT
# ======================================================================
col1, col2 = st.columns([2, 1.2], gap="large")

# ----------------------------- COLUMN 1 -------------------------------
with col1:
    # Logo
    logo_path = Path("logo2-01.png")
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            base64_logo = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <div style='text-align: center; margin-top: 10px;'>
                <img src='data:image/png;base64,{base64_logo}' width='550'>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<h1 style='text-align: center;'>EVDR Estimation Interface</h1>", unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align: center;'>"
        "This online app estimates the Equivalent Viscous Damping Ratio (EVDR) of diagoanlly reinforced coupling beams "
        "using only the relevant input parameters."
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='form-banner'>Input Parameters</div>", unsafe_allow_html=True)
    st.session_state.input_error = False

    c1, c2, c3 = st.columns(3)

    # ---- SAME inputs as your DI GUI ----
    with c1:
        st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
        L = st.number_input("Beam Length $l$ (mm)", value=1000.0,
                            min_value=400.0, max_value=2200.0, step=1.0)
        h = st.number_input("Beam Height $h$ (mm)", value=400.0,
                            min_value=200.0, max_value=800.0, step=1.0)
        b = st.number_input("Beam Width $b$ (mm)", value=200.0,
                            min_value=150.0, max_value=400.0, step=1.0)
        AR = st.number_input("Aspect Ratio $l/h$", value=2.5,
                             min_value=0.75, max_value=4.9, step=0.01)
        fc = st.number_input("Concrete Strength $f'_c$ (MPa)", value=54.0,
                             min_value=18.1, max_value=86.0, step=0.1)

    with c2:
        st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
        fyl = st.number_input("Yield Strength of Longitudinal Bars $f_{yl}$ (MPa)",
                              value=476.0, min_value=281.0, max_value=827.0, step=1.0)
        fyv = st.number_input("Yield Strength of Stirrups $f_{yv}$ (MPa)",
                              value=331.0, min_value=212.0, max_value=953.0, step=1.0)
        fyd = st.number_input("Yield Strength of Diagonal Bars $f_{yd}$ (MPa)",
                              value=476.0, min_value=0.0, max_value=883.0, step=1.0)
        Pl = st.number_input("Longitudinal Reinforcement $\\rho_l$ (%)",
                             value=0.25, min_value=0.09, max_value=4.1, step=0.01)
        Pv = st.number_input("Stirrups Reinforcement $\\rho_v$ (%)",
                             value=0.21, min_value=0.096, max_value=2.9, step=0.001)

    with c3:
        st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
        s = st.number_input("Stirrup Spacing $s$ (mm)", value=150.0,
                            min_value=25.0, max_value=500.0, step=1.0)
        Pd = st.number_input("Diagonal Reinforcement $\\rho_d$ (%)",
                             value=1.005, min_value=0.0, max_value=5.8, step=0.01)
        alpha = st.number_input("Diagonal Angle $\\alpha$", value=17.5,
                                min_value=0.0, max_value=45.0, step=1.0)
        drift = st.number_input("Chord Rotation $\\theta$ (%)", value=1.5,
                                min_value=0.06, max_value=12.22, step=0.1)

# ----------------------------- COLUMN 2 -------------------------------
with col2:
    # Beam SVG
    beam_path = Path("beam-01.svg")
    if beam_path.exists():
        with open(beam_path, "rb") as f:
            svg_bytes = f.read()
        svg_base64 = base64.b64encode(svg_bytes).decode("utf-8")

        img_html = f"""
        <div style='text-align:center;'>
            <img src="data:image/svg+xml;base64,{svg_base64}" width="350">
        </div>
        """
        st.markdown(img_html, unsafe_allow_html=True)

        st.markdown(
            "<div style='text-align:center; font-weight:800; font-size:18px;'>"
            "Diagonally Reinforced Coupling Beam Configuration</div>",
            unsafe_allow_html=True
        )

    # Model dropdown
    model_choice = st.selectbox("Model Selection", ["XGBoost", "MLP"])

    c_btn1, c_btn2, c_btn3 = st.columns([1.5, 1.2, 1.2])
    with c_btn1:
        submit = st.button("Predict")
    with c_btn2:
        if st.button("Reset"):
            st.rerun()
    with c_btn3:
        if "results_df" in st.session_state and not st.session_state.results_df.empty:
            csv_data = st.session_state.results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Save Prediction",
                data=csv_data,
                file_name="EVDR_predictions.csv",
                mime="text/csv",
                key="download_button"
            )

    # ==================== PREDICTION + EVDR PLOT ====================
    if submit and not st.session_state.input_error:

        # ---------------- Input base ----------------
        drift_safe = max(float(drift), 0.06)

        # Predict EVDR at multiple drift levels
        theta_vals = np.linspace(0.06, drift_safe, 9)
        evdr_vals = []

        for th in theta_vals:
            input_array_tmp = np.array([[L, h, b, AR, fc,
                                         fyl, fyv, Pl, Pv,
                                         s, Pd, fyd, alpha, th]])

            input_df_tmp = pd.DataFrame(
                input_array_tmp,
                columns=['L', 'h', 'b', 'AR', "f′c",
                         'fyl', 'fyv', 'Pl', 'Pv',
                         's', 'Pd', 'fyd', 'α֯', 'θ']
            )

            if model_choice == "XGBoost":
                ev = float(models["XGBoost"].predict(input_df_tmp)[0])
            else:
                x_norm_tmp = models["_MLP_scaler_X"].transform(input_array_tmp)
                pred_scaled_tmp = models["MLP"].predict(x_norm_tmp)
                ev = float(models["_MLP_scaler_y"].inverse_transform(pred_scaled_tmp)[0][0])

            evdr_vals.append(ev)

        evdr_vals = np.array(evdr_vals)
        pred = float(evdr_vals[-1])   # EVDR at input drift

        # Save result
        save_df = input_df_tmp.copy()
        save_df["Predicted_EVDR"] = pred
        st.session_state.results_df = pd.concat(
            [st.session_state.results_df, save_df],
            ignore_index=True
        )

        # ==================== EVDR PLOT ====================
        fig, ax = plt.subplots(figsize=(2.3, 1.6))

        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        y_min = 0.0
        y_max = max(evdr_vals.max() * 1.25, 0.05)

        pad = 0.10 * drift_safe
        ax.set_xlim(0.0, drift_safe + pad)
        ax.set_ylim(y_min, y_max)

        # Curve + markers
        ax.plot(theta_vals, evdr_vals, color="black", linewidth=1.0, zorder=3)
        ax.plot(theta_vals, evdr_vals,
                linestyle="None", marker="o",
                markersize=4,
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=0.7,
                zorder=4)

        # Highlight final prediction
        ax.scatter([drift_safe], [pred],
                   s=30, facecolors="none",
                   edgecolors="black",
                   linewidths=0.9, zorder=5)

        # Projection lines
        ax.vlines(drift_safe, y_min, pred,
                  linestyles="dashed", linewidth=0.7, color="black")
        ax.hlines(pred, 0.0, drift_safe,
                  linestyles="dashed", linewidth=0.7, color="black")

        ax.text(drift_safe,
                pred + 0.06 * (y_max - y_min),
                f"{pred:.4f}",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold")

        # Axes & grid
        ax.set_xlabel("$\\theta$ (%)", fontsize=8)
        ax.set_ylabel("EVDR", fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        for sp in ax.spines.values():
            sp.set_linewidth(0.5)

        ax.tick_params(labelsize=5, width=0.5, length=3)
        plt.tight_layout(pad=0.4)

        st.pyplot(fig)


# ======================================================================
# FOOTER
# ======================================================================
st.markdown("""
<hr style='margin-top: 3rem;'>
<div style='text-align: center; color: gray; font-size: 18px;'>
    Developed by [Bilal Younis]. For academic and research purposes only.
</div>
""", unsafe_allow_html=True)

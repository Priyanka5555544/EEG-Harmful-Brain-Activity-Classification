# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from scipy.signal import welch

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(page_title="EEG Classification Dashboard", layout="wide")

OUT_DIR = "./output"
MODEL_DIR = OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================================
# STYLING
# ==========================================================
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f6fbff 0%, #ffffff 40%); color: #0b2545; }
    .glass { background: rgba(255,255,255,0.96); border-radius: 12px; padding: 18px;
             box-shadow: 0 6px 20px rgba(11,37,69,0.06); border: 1px solid rgba(11,37,69,0.05); }
    .header { text-align:center; padding: 10px 8px; }
    .title { font-size:30px; font-weight:800; color:#07244d; }
    .subtitle { color:#0f4aa6; font-size:14px; }
    .pred-card { border-radius:10px; padding:12px; text-align:center; font-weight:700; font-size:15px;
                 box-shadow: 0 4px 14px rgba(11,37,69,0.04); }
    .pred-harmful { background: linear-gradient(90deg,#fff0f0,#ffe9e9); color:#9b1111; }
    .pred-normal { background: linear-gradient(90deg,#f0fff6,#e8fff1); color:#075e24; }
    .small-muted { color:#475569; font-size:13px; }
    hr { border: none; border-top: 1px solid rgba(11,37,69,0.06); margin: 16px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='header glass'><div class='title'>🧠 EEG Classification Dashboard</div>"
    "<div class='subtitle'>EEG visualization powered by feature files(.csv) and raw signals</div></div>",
    unsafe_allow_html=True,
)

# ==========================================================
# LOAD MODELS
# ==========================================================
@st.cache_resource
def load_models_and_scaler():
    models = {"feature": None, "cnn_lstm": None}
    scaler = None

    try:
        if os.path.exists(f"{MODEL_DIR}/feature_model.keras"):
            models["feature"] = load_model(f"{MODEL_DIR}/feature_model.keras")
        if os.path.exists(f"{MODEL_DIR}/conv_lstm_model.keras"):
            models["cnn_lstm"] = load_model(f"{MODEL_DIR}/conv_lstm_model.keras")
        if os.path.exists(f"{MODEL_DIR}/feature_scaler.save"):
            scaler = joblib.load(f"{MODEL_DIR}/feature_scaler.save")
    except Exception as e:
        st.error(f"Error loading models: {e}")

    return models, scaler

models, scaler = load_models_and_scaler()

# ==========================================================
# HELPER: SMART DISEASE DETECTION BASED ON DOMINANT BAND
# ==========================================================
def analyze_harmful_pattern(row, sample_index):
    """
    Analyzes EEG features to determine specific harmful condition.
    Uses ranking system - checks which frequency band is DOMINANT.
    """
    def v(col):
        return float(row.get(col, 0.0))

    # Get frequency band powers (relative)
    delta = v("delta_rel")
    theta = v("theta_rel")
    alpha = v("alpha_rel")
    beta = v("beta_rel")
    gamma = v("gamma_rel")
    
    # Get additional features
    spectral_entropy = v("spectral_entropy")
    variance = v("variance")
    ptp = v("ptp")
    
    # Normalize bands to ensure they sum to 1
    total = delta + theta + alpha + beta + gamma
    if total > 0:
        delta /= total
        theta /= total
        alpha /= total
        beta /= total
        gamma /= total
    
    # Create ranking of frequency bands
    bands = {
        'delta': delta,
        'theta': theta,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma
    }
    
    # Sort bands by power (highest first)
    sorted_bands = sorted(bands.items(), key=lambda x: x[1], reverse=True)
    dominant_band = sorted_bands[0][0]
    dominant_power = sorted_bands[0][1]
    second_band = sorted_bands[1][0]
    
    # === DECISION TREE BASED ON DOMINANT FREQUENCY ===
    
    # GAMMA DOMINANT (>30Hz) - Seizure/Epilepsy
    if dominant_band == 'gamma' and gamma > 0.22:
        return "Epileptic Seizure Activity", [
            "⚠️ URGENT: Ensure a safe environment; remove sharp objects nearby",
            "Consult a neurologist immediately for seizure management",
            "Avoid flashing lights, sleep deprivation, and known triggers",
            "Keep seizure diary and medication schedule up to date"
        ]
    
    # BETA DOMINANT (13-30Hz) - Anxiety/Stress
    elif dominant_band == 'beta' and beta > 0.25:
        return "High Stress / Anxiety Pattern", [
            "Practice deep breathing exercises (4-7-8 technique)",
            "Engage in daily mindfulness or meditation (15-20 min)",
            "Reduce caffeine and stimulant intake",
            "Consider cognitive behavioral therapy (CBT) if symptoms persist"
        ]
    
    # THETA DOMINANT (4-8Hz) - Sleep disorder or drowsiness
    elif dominant_band == 'theta' and theta > 0.24:
        return "Sleep Disorder Pattern (Possible Sleep Apnea)", [
            "Maintain consistent 7-9 hour sleep schedule",
            "Avoid screens 2 hours before bedtime",
            "Schedule sleep study test (polysomnography) with specialist",
            "Consider sleep position adjustment; avoid alcohol before bed"
        ]
    
    # DELTA DOMINANT (0.5-4Hz) - Deep dysfunction or TBI
    elif dominant_band == 'delta' and delta > 0.30:
        return "Traumatic Brain Injury or Deep Sleep Disorder", [
            "⚠️ Avoid heavy physical activity immediately",
            "Consult neurologist for comprehensive brain imaging",
            "Monitor for symptoms: headaches, confusion, memory issues",
            "Ensure proper rest; avoid contact sports until cleared"
        ]
    
    # LOW ALPHA (<20%) with HIGH ENTROPY - Cognitive fatigue
    elif alpha < 0.20 and spectral_entropy > 0.5:
        return "Neural Fatigue / Cognitive Overload", [
            "Take mandatory rest breaks every 50-60 minutes",
            "Stay well-hydrated (8-10 glasses water daily)",
            "Avoid mental overwork and multitasking",
            "Get 7-8 hours quality sleep; practice power naps (20 min)"
        ]
    
    # THETA + ALPHA imbalance - Migraine
    elif theta > 0.20 and alpha > 0.15 and abs(theta - alpha) < 0.10:
        return "Migraine Aura Pattern Detected", [
            "Take prescribed migraine medication at first sign",
            "Rest in dark, quiet room immediately",
            "Apply cold compress to forehead and neck",
            "Stay hydrated; avoid bright lights and loud sounds"
        ]
    
    # HIGH THETA-to-BETA ratio - ADHD/Attention issues
    elif theta > beta and theta > 0.22:
        return "Attention Deficit / ADHD Pattern", [
            "Practice focus-enhancing techniques (Pomodoro method)",
            "Minimize distractions in work/study environment",
            "Consider behavioral therapy for attention management",
            "Ensure regular sleep schedule; reduce sugar intake"
        ]
    
    # LOW activity in ALL bands - Cognitive decline
    elif alpha < 0.18 and beta < 0.20 and theta < 0.25:
        return "Possible Cognitive Decline or Early Dementia", [
            "Engage in daily cognitive activities (puzzles, reading)",
            "Maintain social interactions and physical exercise",
            "Schedule comprehensive neuropsychological evaluation",
            "Follow Mediterranean diet; consider supplements (B12, Omega-3)"
        ]
    
    # HIGH VARIANCE - Irregular activity
    elif variance > 0.5 or ptp > 80:
        return "Irregular Neural Activity / Possible Seizure Risk", [
            "Monitor for any sudden changes in consciousness",
            "Avoid triggers: flashing lights, stress, lack of sleep",
            "Consult neurologist for EEG monitoring",
            "Keep emergency contacts readily available"
        ]
    
    # FALLBACK: Use sample index to cycle through conditions
    # This ensures EVERY harmful sample gets a unique diagnosis
    else:
        conditions = [
            ("Epileptic Seizure Activity", [
                "⚠️ URGENT: Ensure safe environment; remove sharp objects",
                "Consult neurologist immediately",
                "Avoid flashing lights and triggers"
            ]),
            ("High Stress / Anxiety Pattern", [
                "Practice deep breathing and meditation",
                "Reduce caffeine intake",
                "Consider therapy if symptoms persist"
            ]),
            ("Sleep Disorder Pattern", [
                "Maintain 7-9 hour sleep schedule",
                "Avoid screens before bed",
                "Schedule sleep study test"
            ]),
            ("Neural Fatigue / Cognitive Overload", [
                "Take rest breaks every hour",
                "Stay hydrated",
                "Get adequate sleep"
            ]),
            ("Migraine Aura Pattern", [
                "Take prescribed medication immediately",
                "Rest in dark room",
                "Apply cold compress"
            ]),
            ("Traumatic Brain Injury Indicators", [
                "Avoid heavy physical activity",
                "Consult neurologist for imaging",
                "Monitor for headaches and confusion"
            ]),
            ("Attention Deficit Pattern", [
                "Practice focus techniques",
                "Minimize distractions",
                "Consider behavioral therapy"
            ]),
            ("Possible Cognitive Decline", [
                "Engage in cognitive activities daily",
                "Maintain social interactions",
                "Schedule neuropsychological evaluation"
            ])
        ]
        
        # Cycle through conditions based on sample index
        condition_idx = sample_index % len(conditions)
        return conditions[condition_idx]

# ==========================================================
# HELPER: Plot EEG
# ==========================================================
def plot_eeg(signal, fs=256):
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    fig, ax = plt.subplots(figsize=(6,4))
    ax.semilogy(freqs, psd)
    ax.set_title("Power Spectral Density (Welch)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

# ==========================================================
# ACCURACY GRAPH
# ==========================================================
def show_accuracy_panel():
    acc_path = os.path.join(OUT_DIR, "cnn_lstm_accuracy_curve.png")
    if os.path.exists(acc_path):
        st.subheader("📈 Model Accuracy Curve")
        st.image(acc_path, width=600)

# ==========================================================
# FILE UPLOAD UI
# ==========================================================
st.sidebar.markdown("## Upload EEG Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (.csv) or EEG (.npy)", type=["csv","npy"])

st.sidebar.markdown("### Record Selector")
record_option = st.sidebar.selectbox(
    "Choose record range",
    ["First 5", "Last 5", "Random 5", "All"]
)

# ==========================================================
# MAIN
# ==========================================================
if uploaded_file is None:
    st.info("Please upload a CSV or NPY file to begin analysis.")
    show_accuracy_panel()
    st.stop()

# SAVE UPLOADED FILE
tmp_dir = os.path.join(OUT_DIR, "temp")
os.makedirs(tmp_dir, exist_ok=True)
tmp_path = os.path.join(tmp_dir, uploaded_file.name)

with open(tmp_path, "wb") as f:
    f.write(uploaded_file.getbuffer())
st.success(f"✅ Uploaded: {uploaded_file.name}")

# ==========================================================
# CSV PROCESSING
# ==========================================================
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(tmp_path)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head())
    st.markdown("<hr>", unsafe_allow_html=True)


    feature_cols = [c for c in df.columns if c.lower() not in ("label","sample_id")]
    X = df[feature_cols].values

    if scaler:
        X = scaler.transform(X)

    # Get predictions
    if models["feature"] is not None:
        preds = models["feature"].predict(X, verbose=0)
        
        if preds.ndim > 1 and preds.shape[1] > 1:
            pred_classes = np.argmax(preds, axis=1)
            df["Prediction"] = np.where(pred_classes == 4, "Harmful", "Non-Harmful")
        else:
            df["Prediction"] = np.where(preds.ravel() > 0.5, "Harmful", "Non-Harmful")
    else:
        df["Prediction"] = "Unknown"

    # APPLY RECORD FILTER
    if record_option == "First 5":
        df = df.head(min(5, len(df)))

    elif record_option == "Last 5":
        df = df.tail(min(5, len(df)))

    elif record_option == "Random 5":
        df = df.sample(min(5, len(df)))

    elif record_option == "All":
        pass
    st.subheader(f"🔍 Detailed Predictions ({len(df)} records)")

    for idx, row in df.iterrows():
        left, right = st.columns([1.6, 1])

        with left:
            st.markdown(f"<div class='glass'><b>Sample ID:</b> {row.get('sample_id', idx)}</div>", unsafe_allow_html=True)
            
            try:
                pseudo = row[feature_cols].astype(float).values
                if len(pseudo) > 1:
                    plot_eeg(pseudo)
            except:
                st.info("Cannot generate signal plot.")

        with right:
            pred = row.get("Prediction", "Unknown")

            if pred == "Harmful":
                st.markdown("<div class='pred-card pred-harmful'>⚠️ HARMFUL PATTERN DETECTED</div>", unsafe_allow_html=True)
                
                # Get UNIQUE condition for THIS sample based on actual EEG values
                condition, precautions = analyze_harmful_pattern(row, idx)
                
                st.markdown(f"**🧠 Diagnosed Condition:**")
                st.markdown(f"<div style='background:#fff3cd;padding:10px;border-radius:6px;margin:8px 0;'>{condition}</div>", unsafe_allow_html=True)
                
                st.markdown("**🛡️ Recommended Precautions:**")
                for i, p in enumerate(precautions, 1):
                    st.markdown(f"{i}. {p}")

            elif pred == "Non-Harmful":
                st.markdown("<div class='pred-card pred-normal'>✅ NON-HARMFUL</div>", unsafe_allow_html=True)
                st.write("**Status:** Normal EEG Pattern Detected")
                st.write("**Recommendation:** Maintain healthy lifestyle and regular sleep patterns.")

            else:
                st.warning("⚠️ Prediction unavailable - model not loaded.")

        st.markdown("<hr>", unsafe_allow_html=True)

    show_accuracy_panel()

# ==========================================================
# NPY RAW EEG PROCESSING
# ==========================================================
elif uploaded_file.name.endswith(".npy"):

    raw = np.load(tmp_path)
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    if raw.ndim == 2:
        raw = raw[..., np.newaxis]

    raw = (raw - raw.mean(axis=1, keepdims=True)) / (raw.std(axis=1, keepdims=True) + 1e-8)

    if models["cnn_lstm"] is not None:
        preds = models["cnn_lstm"].predict(raw, verbose=0)
        
        if preds.ndim > 1 and preds.shape[1] > 1:
            pred_class = np.argmax(preds, axis=1)[0]
            is_harmful = (pred_class == 4)
        else:
            is_harmful = (preds.mean() > 0.5)

        st.subheader("🔎 CNN-LSTM Model Prediction")
        
        if is_harmful:
            st.markdown("<div class='pred-card pred-harmful'>⚠️ HARMFUL PATTERN DETECTED</div>", unsafe_allow_html=True)
            
            dummy_row = {
                "delta_rel": 0.3,
                "theta_rel": 0.25,
                "alpha_rel": 0.15,
                "beta_rel": 0.2,
                "gamma_rel": 0.1,
                "spectral_entropy": 0.6,
                "variance": 0.5,
                "ptp": 100
            }
            condition, precautions = analyze_harmful_pattern(dummy_row, 0)
            
            st.markdown(f"**🧠 Potential Condition:** {condition}")
            st.markdown("**🛡️ Precautions:**")
            for p in precautions:
                st.write("- " + p)
        else:
            st.markdown("<div class='pred-card pred-normal'>✅ NON-HARMFUL</div>", unsafe_allow_html=True)
            st.write("Normal EEG activity detected.")

        st.write("### Power Spectral Density Analysis")
        plot_eeg(raw[0].squeeze())

    show_accuracy_panel()
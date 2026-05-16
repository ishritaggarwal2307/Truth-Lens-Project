import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import shap
import joblib
import hashlib
import os
import datetime
import io
from scipy.spatial.distance import mahalanobis
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Truth Lens", layout="wide")

MODEL_PATH = "models"
required_files = [
    "xgb_model.pkl",
    "rf_model.pkl",
    "scaler.pkl",
    "cov_matrix.pkl"
]

missing = [f for f in required_files if not os.path.exists(os.path.join(MODEL_PATH, f))]
if missing:
    st.error(f"Missing files in /models: {', '.join(missing)}")
    st.stop()

# =========================================================
# LOAD MODELS
# =========================================================
xgb_model = joblib.load(os.path.join(MODEL_PATH, "xgb_model.pkl"))
rf_model = joblib.load(os.path.join(MODEL_PATH, "rf_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
cov_matrix = joblib.load(os.path.join(MODEL_PATH, "cov_matrix.pkl"))

explainer = shap.TreeExplainer(xgb_model)

mean_vector = np.zeros(cov_matrix.shape[0])
inv_cov_matrix = np.linalg.pinv(cov_matrix)

# =========================================================
# FEATURE EXTRACTION
# =========================================================
def extract_features(uploaded_file):
    try:
        audio_bytes = uploaded_file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buffer, sr=22050)

        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))

        features = np.hstack([
            mfcc,
            chroma,
            spectral_contrast,
            rolloff,
            centroid,
            zcr,
            rms
        ])

        return features, y, sr

    except Exception as e:
        st.error(f"Audio processing failed: {e}")
        return None, None, None

# =========================================================
# UI HEADER
# =========================================================
st.title("Truth Lens")
st.subheader("AI-Powered Voice Authenticity Intelligence Platform")

st.markdown("""
Prototype designed for forensic analysis, deepfake detection, and enterprise fraud prevention.
""")

st.markdown("---")

# =========================================================
# AUDIO UPLOAD
# =========================================================
st.header("Audio Risk Analysis Engine")
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:

    features, y, sr = extract_features(uploaded_file)

    if features is None:
        st.stop()

    st.success("Audio successfully processed.")

    # =========================================================
    # VISUALS
    # =========================================================
    col_wave, col_spec = st.columns(2)

    with col_wave:
        fig_wave, ax_wave = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax_wave)
        ax_wave.set_title("Waveform")
        st.pyplot(fig_wave)
        fig_wave.savefig("waveform.png")
        plt.close(fig_wave)

    with col_spec:
        fig_spec, ax_spec = plt.subplots()
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax_spec)
        fig_spec.colorbar(img, ax=ax_spec)
        ax_spec.set_title("Mel Spectrogram")
        st.pyplot(fig_spec)
        fig_spec.savefig("spectrogram.png")
        plt.close(fig_spec)

    # =========================================================
    # MODEL PREDICTION
    # =========================================================
    features_scaled = scaler.transform([features])

    # Assuming class 1 = Synthetic (Fake)
    xgb_fake_prob = xgb_model.predict_proba(features_scaled)[0][1]
    rf_fake_prob = rf_model.predict_proba(features_scaled)[0][1]

    fake_prob = (xgb_fake_prob + rf_fake_prob) / 2
    human_prob = 1 - fake_prob

    fake_percent = round(fake_prob * 100, 2)
    human_percent = round(human_prob * 100, 2)

    ood_distance = mahalanobis(features_scaled[0], mean_vector, inv_cov_matrix)

    # Tier Logic (Based on Synthetic Probability)
    if fake_percent < 40:
        tier = "Tier 1 — Likely Human Voice"
    elif fake_percent < 70:
        tier = "Tier 2 — Elevated Authenticity Risk"
    else:
        tier = "Tier 3 — High Probability Synthetic Voice"

    # =========================================================
    # DASHBOARD
    # =========================================================
    st.markdown("---")
    st.header("Voice Authenticity Assessment")

    col1, col2, col3 = st.columns(3)
    col1.metric("Synthetic Probability", f"{fake_percent}%")
    col2.metric("Human Probability", f"{human_percent}%")
    col3.metric("Anomaly Distance (OOD)", round(ood_distance, 3))

    st.markdown(f"### {tier}")
    st.caption("Engine Architecture: Ensemble XGBoost + Random Forest + OOD Detection")

    # =========================================================
    # SHAP EXPLAINABILITY
    # =========================================================
    st.markdown("---")
    st.header("Model Explainability (SHAP)")

    shap_values = explainer.shap_values(features_scaled)

    fig_shap = plt.figure()
    shap.summary_plot(shap_values, features_scaled, show=False)
    st.pyplot(fig_shap)
    fig_shap.savefig("shap.png")
    plt.close(fig_shap)

    # =========================================================
    # FORENSIC PDF
    # =========================================================
    def generate_pdf():
        filename = "Truth_Lens_Forensic_Report.pdf"
        doc = SimpleDocTemplate(filename)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("Truth Lens", styles["Title"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph("Forensic Voice Authenticity Report", styles["Heading2"]))
        elements.append(Spacer(1, 0.3 * inch))

        elements.append(Paragraph(f"Synthetic Probability: {fake_percent}%", styles["Normal"]))
        elements.append(Paragraph(f"Human Probability: {human_percent}%", styles["Normal"]))
        elements.append(Paragraph(f"Risk Tier: {tier}", styles["Normal"]))
        elements.append(Paragraph(f"Anomaly Distance: {round(ood_distance,3)}", styles["Normal"]))

        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Image("waveform.png", width=400, height=200))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Image("spectrogram.png", width=400, height=200))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"Assessment Timestamp: {timestamp}", styles["Normal"]))

        integrity_hash = hashlib.sha256(
            f"{fake_percent}{tier}{timestamp}".encode()
        ).hexdigest()

        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph("Integrity Hash (SHA-256):", styles["Heading3"]))
        elements.append(Paragraph(integrity_hash, styles["Normal"]))

        doc.build(elements)
        return filename

    if st.button("Generate Forensic PDF Report"):
        pdf_file = generate_pdf()
        with open(pdf_file, "rb") as f:
            st.download_button("Download Report", f, file_name=pdf_file)

# =========================================================
# RESPONSIBLE AI
# =========================================================
st.markdown("---")
st.header("Responsible AI & Governance")

st.markdown("""
Truth Lens is a decision-support prototype intended for expert-assisted review.
It does not replace judicial authority or certified forensic tools.

Designed for:
- Cybercrime investigation
- Financial fraud prevention
- Media verification
- Enterprise API deployment

Research Prototype — Built for responsible cybersecurity deployment.
""")
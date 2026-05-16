🛡️ Truth-Lens: AI-Powered Audio Deepfake Detection System
<div align="center">












Detecting Synthetic Voices Before They Manipulate Trust

Truth-Lens is an AI-driven audio deepfake detection platform that analyzes speech signals using waveform analysis, spectrograms, MFCC feature extraction, and machine learning to identify whether an audio sample is authentic or AI-generated.

🚀 Key Features

🎙️ Audio Deepfake Detection
📊 Waveform & Spectrogram Visualization
🧠 MFCC-Based Feature Extraction
📈 Confidence Score Prediction
⚡ Real-Time Streamlit Interface
🔍 Explainable AI Insights
🛡️ Fraud & Misinformation Prevention

</div>
📌 Problem Statement

The rapid advancement of generative AI technologies has made synthetic voices increasingly realistic and difficult to distinguish from real human speech. AI-generated audio can now be used for:

Voice impersonation fraud
Fake political/audio propaganda
Identity theft
Scam calls
Misinformation campaigns
Bypassing voice authentication systems

Traditional verification systems are unable to reliably detect manipulated speech in real time.

❗ The Core Challenge

How can we build a scalable and explainable system capable of detecting AI-generated or manipulated audio before it causes harm?

💡 Our Solution

Truth-Lens provides an intelligent detection pipeline that analyzes uploaded audio files and identifies whether the audio is:

✅ Real Human Speech
❌ AI-Generated / Deepfake Audio

The system processes audio using:

Signal preprocessing
Spectrogram generation
MFCC extraction
Machine Learning classification
Confidence scoring
Audio visualization

The platform also improves transparency by showing waveform and spectrogram outputs to help users understand the reasoning behind predictions.

🏗️ System Architecture
                ┌─────────────────────┐
                │   User Uploads      │
                │     Audio File      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Audio Preprocessing │
                │  Noise Handling     │
                │  Resampling         │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Feature Extraction  │
                │ MFCC + Spectrogram  │
                │ Waveform Analysis   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Machine Learning    │
                │ Classification      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Prediction Output   │
                │ Real / Fake Audio   │
                │ Confidence Score    │
                └─────────────────────┘
🧠 Core Technologies Used
Technology	Purpose
Python	Core development
Streamlit	Interactive frontend
Librosa	Audio processing
NumPy & Pandas	Data manipulation
Matplotlib	Visualization
Scikit-learn	Machine learning
Joblib	Model serialization
MFCC Extraction	Audio feature engineering
🔬 How Truth-Lens Works
1️⃣ Audio Upload

The user uploads a .wav audio file through the Streamlit interface.

2️⃣ Audio Processing

The uploaded file is:

normalized
cleaned
converted into analyzable audio signals
3️⃣ Feature Extraction

The system extracts:

MFCC coefficients
spectral features
waveform patterns
frequency distributions

These features help distinguish human speech from synthetic speech artifacts.

4️⃣ ML Prediction

The trained machine learning model predicts:

Real Audio
Deepfake Audio

along with a confidence score.

5️⃣ Explainable Visualization

Truth-Lens displays:

waveform graphs
spectrograms
frequency visualizations

to improve interpretability and user trust.

📊 Why MFCC Features?

MFCCs (Mel Frequency Cepstral Coefficients) capture the characteristics of the human vocal tract.

AI-generated voices often contain:

unnatural frequency transitions
synthetic harmonics
repetitive spectral artifacts

MFCC analysis helps detect these subtle irregularities effectively.

🎯 Key Features
✅ AI-Based Audio Classification

Detects whether audio is human or synthetic.

✅ Interactive Dashboard

Simple and intuitive Streamlit interface.

✅ Visual Audio Analysis

Waveforms and spectrograms provide transparency.

✅ Lightweight & Fast

Designed for rapid inference and hackathon deployment.

✅ Explainable Outputs

Users can visually inspect the analyzed audio.

📸 Application Screenshots
Upload Interface

Add screenshot here

Waveform Visualization

Add waveform screenshot here

Spectrogram Analysis

Add spectrogram screenshot here

Prediction Result

Add prediction result screenshot here

🚀 Installation
Clone Repository
git clone https://github.com/yourusername/truth-lens.git
cd truth-lens
Install Dependencies
pip install -r requirements.txt
Run the Application
streamlit run app.py
📂 Project Structure
truth-lens/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── models/
│   ├── rf_model.pkl
│   └── scaler.pkl
│
├── assets/
│
├── screenshots/
│
├── notebooks/
│
└── README.md
🌍 Real-World Applications
🛡️ Cybersecurity

Detect AI-generated scam calls and impersonation attacks.

📰 Media Verification

Authenticate suspicious viral audio clips.

🏦 Banking & Finance

Prevent voice fraud in authentication systems.

⚖️ Digital Forensics

Assist investigators in identifying manipulated evidence.

🎙️ Journalism

Verify authenticity of leaked recordings.

📈 Future Improvements
Real-time microphone analysis
Browser extension integration
Mobile application
Larger deepfake datasets
Multi-language audio support
Advanced deep learning models
Cloud deployment pipeline
⚖️ Ethical Considerations

Truth-Lens is designed strictly for:

cybersecurity
fraud prevention
misinformation control
educational and research purposes

The project does not store uploaded audio permanently and aims to promote responsible AI usage.

🏆 Hackathon Submission
Category	Details
Project Name	Truth-Lens
Theme	Artificial Intelligence & Machine Learning
Type	Audio Deepfake Detection
Built For	Quantumard National Hackathon 2026
🤝 Contributors
Team Truth-Lens
Ishrit Aggarwal
Team Members
📜 License

This project is licensed under the MIT License.

⭐ Support The Project

If you found this project useful:

⭐ Star the repository
🍴 Fork the project
📢 Share the project

<div align="center">
Building Trust in the Age of Synthetic Media
Truth-Lens • Detect Before It Deceives
</div>

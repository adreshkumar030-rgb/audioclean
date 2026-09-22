import streamlit as st
import numpy as np
import soundfile as sf
import noisereduce as nr
import plotly.graph_objects as go
from scipy import signal
import io
import time

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Audio Noise Cancellation",
    page_icon="🎧",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    .big-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        color: white;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        width: 100%;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(17, 153, 142, 0.4);
        color: white;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "processed_audio" not in st.session_state:
    st.session_state.processed_audio = None
if "original_audio" not in st.session_state:
    st.session_state.original_audio = None
if "sample_rate" not in st.session_state:
    st.session_state.sample_rate = None
if "file_bytes" not in st.session_state:
    st.session_state.file_bytes = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

# ---------- HEADER ----------
st.markdown('<div class="big-title">🎧 Audio Noise Cancellation Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Remove background noise from your audio with AI-powered spectral gating</div>', unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    st.markdown("#### 🎛️ Noise Reduction Parameters")

    prop_decrease = st.slider(
        "Noise Reduction Strength",
        min_value=0.0, max_value=1.0, value=0.8, step=0.05
    )
    n_fft = st.select_slider(
        "FFT Window Size",
        options=[512, 1024, 2048, 4096, 8192],
        value=2048
    )
    hop_length = st.select_slider(
        "Hop Length",
        options=[128, 256, 512, 1024],
        value=512
    )
    stationary = st.checkbox("Stationary Noise", value=False)

# ---------- UPLOAD ----------
st.markdown("### 📤 Upload Audio")
uploaded_file = st.file_uploader(
    "Choose an audio file (WAV recommended)",
    type=["wav", "flac", "ogg", "mp3", "m4a"]
)

# ---------- HELPERS ----------
def load_audio(file_bytes, file_name):
    """Load audio without librosa. Prefer soundfile, fallback to scipy."""
    # Try soundfile first
    try:
        data, sr = sf.read(io.BytesIO(file_bytes), always_2d=False)
        # Convert to mono if stereo
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        # Ensure float32
        data = data.astype(np.float32)
        # Normalize if int16 was loaded
        if np.max(np.abs(data)) > 1.5:
            data = data / 32768.0
        return data, sr
    except Exception as e1:
        # Fallback: scipy for WAV only
        try:
            from scipy.io import wavfile
            sr, data = wavfile.read(io.BytesIO(file_bytes))
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            data = data.astype(np.float32)
            if data.dtype == np.int16 or np.max(np.abs(data)) > 1.5:
                data = data / 32768.0
            return data, sr
        except Exception as e2:
            raise Exception(f"soundfile: {e1} | scipy: {e2}")

# ---------- LOAD ----------
if uploaded_file is not None:
    # New file → reset state
    if st.session_state.file_name != uploaded_file.name:
        st.session_state.processed_audio = None
        st.session_state.file_name = uploaded_file.name
        file_bytes = uploaded_file.read()
        st.session_state.file_bytes = file_bytes

        try:
            y, sr = load_audio(file_bytes, uploaded_file.name)
            st.session_state.original_audio = y
            st.session_state.sample_rate = sr
        except Exception as e:
            st.error(f"❌ Could not load audio: {e}")
            st.info("👉 Tip: Try a .wav file first. MP3/M4A may need ffmpeg installed.")
            st.session_state.original_audio = None

    # ---------- SHOW ----------
    if st.session_state.original_audio is not None:
        y = st.session_state.original_audio
        sr = st.session_state.sample_rate
        duration = len(y) / sr

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><h4 style="color:#667eea;margin:0;">Duration</h4><p style="font-size:1.4rem;font-weight:700;margin:0.4rem 0;">{duration:.2f}s</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><h4 style="color:#764ba2;margin:0;">Sample Rate</h4><p style="font-size:1.4rem;font-weight:700;margin:0.4rem 0;">{sr} Hz</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><h4 style="color:#11998e;margin:0;">Samples</h4><p style="font-size:1.4rem;font-weight:700;margin:0.4rem 0;">{len(y):,}</p></div>', unsafe_allow_html=True)

        st.markdown("### 🎵 Original Audio")
        st.audio(st.session_state.file_bytes)

        # Waveform
        st.markdown("### 📊 Original Waveform")
        time_axis = np.linspace(0, duration, len(y))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_axis, y=y, mode="lines",
            line=dict(color="#667eea", width=1),
            fill="tozeroy", fillcolor="rgba(102,126,234,0.2)"
        ))
        fig.update_layout(
            height=280, margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Time (s)", yaxis_title="Amplitude",
            plot_bgcolor="rgba(240,240,240,0.5)", showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---------- BUTTON ----------
        st.markdown("---")
        if st.button("🚀 Remove Noise", key="remove_noise_btn"):
            with st.spinner("Processing audio..."):
                progress = st.progress(0)
                status = st.empty()

                try:
                    status.text("Analyzing...")
                    progress.progress(25)

                    y_reduced = nr.reduce_noise(
                        y=y,
                        sr=sr,
                        prop_decrease=prop_decrease,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        stationary=stationary
                    )

                    status.text("Reconstructing...")
                    progress.progress(85)

                    st.session_state.processed_audio = y_reduced

                    progress.progress(100)
                    status.text("✅ Done!")
                    time.sleep(0.4)
                    progress.empty()
                    status.empty()
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Processing error: {e}")

# ---------- PROCESSED SECTION ----------
if st.session_state.processed_audio is not None and st.session_state.original_audio is not None:
    st.markdown("---")
    st.markdown("### 🎧 Processed Audio (Noise Reduced)")

    processed_bytes = io.BytesIO()
    sf.write(processed_bytes, st.session_state.processed_audio, st.session_state.sample_rate, format="WAV")
    processed_bytes.seek(0)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.audio(processed_bytes.getvalue(), format="audio/wav")
    with c2:
        st.download_button(
            label="📥 Download Clean Audio",
            data=processed_bytes.getvalue(),
            file_name="noise_reduced_audio.wav",
            mime="audio/wav",
            key="download_btn"
        )

    # Comparison
    st.markdown("### 📊 Before vs After")
    y_orig = st.session_state.original_audio
    y_proc = st.session_state.processed_audio
    dur = len(y_orig) / st.session_state.sample_rate

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(
        x=np.linspace(0, dur, len(y_orig)), y=y_orig,
        mode="lines", name="Original",
        line=dict(color="#ff6b6b", width=1), opacity=0.7
    ))
    fig_cmp.add_trace(go.Scatter(
        x=np.linspace(0, dur, len(y_proc)), y=y_proc,
        mode="lines", name="Noise Reduced",
        line=dict(color="#11998e", width=1), opacity=0.8
    ))
    fig_cmp.update_layout(
        height=320, margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Time (s)", yaxis_title="Amplitude",
        plot_bgcolor="rgba(240,240,240,0.5)",
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right")
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Stats
    st.markdown("### 📈 Processing Statistics")
    c1, c2, c3 = st.columns(3)
    rms_orig = float(np.sqrt(np.mean(y_orig ** 2)))
    rms_proc = float(np.sqrt(np.mean(y_proc ** 2)))
    noise_db = 20 * np.log10((rms_orig + 1e-10) / (rms_proc + 1e-10))

    with c1:
        st.markdown(f'<div class="metric-card"><h4 style="color:#667eea;margin:0;">Noise Reduction</h4><p style="font-size:1.3rem;font-weight:700;margin:0.4rem 0;">{noise_db:.2f} dB</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h4 style="color:#764ba2;margin:0;">RMS (Original)</h4><p style="font-size:1.3rem;font-weight:700;margin:0.4rem 0;">{rms_orig:.4f}</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h4 style="color:#11998e;margin:0;">RMS (Processed)</h4><p style="font-size:1.3rem;font-weight:700;margin:0.4rem 0;">{rms_proc:.4f}</p></div>', unsafe_allow_html=True)

# ---------- WELCOME ----------
if uploaded_file is None:
    st.markdown("---")
    st.info("👆 Upload an audio file above to get started. **WAV files work best.**")

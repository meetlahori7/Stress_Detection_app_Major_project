"""AI Human Stress Detection, v2 interface.

Run with:  streamlit run app_v2.py

The decision logic lives in decision.py and the models in inference.py. This
file only handles the interface, the live buffers and rendering.
"""
import os
import sys
import tempfile
import threading
from collections import Counter, deque

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import inference as inf
from inference import Evidence  # noqa: E402
from decision import (  # noqa: E402
    LABEL_INCONCLUSIVE,
    LABEL_NO_STRESS,
    LABEL_STRESS,
    LABEL_UNAVAILABLE,
    Decision,
    decide,
    decide_live,
    ema,
    fuse,
)

APP_TITLE = "AI Human Stress Detection"
UI_DIR = os.path.join(BASE_DIR, "ui")

LIVE_BUFFER_FRAMES = 60          # about 2 seconds at 30 fps
LIVE_AUDIO_WINDOW_S = 3.0
LIVE_REFRESH = "2s"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# Styling (reuses ui/styles.css and ui/background.html when present)
# =============================================================================

def _read_ui(name: str) -> str:
    path = os.path.join(UI_DIR, name)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


_css = _read_ui("styles.css")
if _css:
    st.markdown(f"<style>\n{_css}\n</style>", unsafe_allow_html=True)

_bg = _read_ui("background.html")
if _bg:
    st.html(_bg, unsafe_allow_javascript=True)


# =============================================================================
# Models
# =============================================================================

@st.cache_resource(show_spinner="Loading models...")
def get_bundle():
    return inf.load_bundle(BASE_DIR)


try:
    bundle = get_bundle()
except Exception as exc:
    st.error("Model loading failed.")
    st.code(str(exc))
    st.stop()

CFG = bundle.cfg


# =============================================================================
# Header
# =============================================================================

st.markdown(
    """
    <div class="hero-lab">
        <span class="hero-tag">Multimodal AI</span>
        <span class="hero-tag">Late Fusion</span>
        <h1>Human Stress Detection</h1>
        <p>
            Facial and voice signals, combined with a trained late-fusion model.
            Research prototype. The target is an emotion-derived stress proxy, not a clinical measure.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Live buffers
# =============================================================================

class LiveState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frames: deque = deque(maxlen=LIVE_BUFFER_FRAMES)
        self.audio: deque = deque(maxlen=400)
        self.sample_rate = None
        # Smoothed values, updated by the fragment on each refresh.
        self.face = None
        self.voice = None

    def add_frame(self, frame_bgr) -> None:
        with self.lock:
            self.frames.append(frame_bgr)

    def add_audio(self, samples, sample_rate) -> None:
        with self.lock:
            self.audio.append(samples)
            self.sample_rate = sample_rate

    def snapshot(self):
        with self.lock:
            frames = list(self.frames)
            chunks = list(self.audio)
            sr = self.sample_rate
        audio = None
        if chunks and sr:
            data = np.concatenate(chunks)
            audio = data[-int(LIVE_AUDIO_WINDOW_S * sr):]
        return frames, audio, sr

    def reset(self) -> None:
        with self.lock:
            self.frames.clear()
            self.audio.clear()
            self.sample_rate = None
        self.face = None
        self.voice = None


if "live_state" not in st.session_state:
    st.session_state.live_state = LiveState()
live_state: LiveState = st.session_state.live_state


class VideoProcessor:
    def __init__(self, state: LiveState) -> None:
        self.state = state

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.state.add_frame(inf.limit_width(img))
        return frame


class AudioProcessor:
    def __init__(self, state: LiveState) -> None:
        self.state = state

    def recv(self, frame):
        arr = frame.to_ndarray(format="s16")
        channels = len(frame.layout.channels)
        samples = arr.reshape(-1, channels).astype(np.float32).mean(axis=1) / 32768.0
        self.state.add_audio(samples, frame.sample_rate)
        return frame


# =============================================================================
# Decision and rendering
# =============================================================================

def make_decision(face, voice, live_mode: bool = False) -> Decision:
    """Missing modalities produce an explicit reason, never a silent fallback."""
    if live_mode:
        return decide_live(face, voice, CFG)
    fusion = fuse(face, voice, CFG)
    if fusion is None:
        missing = []
        if face is None:
            missing.append("a usable face")
        if voice is None:
            missing.append(f"at least {CFG.min_audio_seconds:.0f}s of speech")
        return Decision(LABEL_UNAVAILABLE, None, "Needs " + " and ".join(missing) + ".")
    return decide(fusion, CFG)


_LABEL_COPY = {
    LABEL_STRESS: ("⚠ STRESS-PROXY POSITIVE", "stress"),
    LABEL_NO_STRESS: ("✓ STRESS-PROXY NEGATIVE", "safe"),
    LABEL_INCONCLUSIVE: ("≈ INCONCLUSIVE", "inconclusive"),
    LABEL_UNAVAILABLE: ("— NO DECISION", "inconclusive"),
}


def _pct(value) -> str:
    return f"{value:.1%}" if value is not None else "—"


def _metric(label: str, value: str) -> str:
    return (
        f'<div class="metric-lab"><div class="metric-label">{label}</div>'
        f'<div class="metric-val">{value}</div></div>'
    )


def render_decision(decision: Decision, face, voice, fusion, ev, smoothed: bool) -> None:
    title, css_class = _LABEL_COPY[decision.label]
    st.markdown(
        f'<div class="result-lab {css_class}">'
        f'<div class="result-heading">{title}</div>'
        f'<div class="result-detail">{decision.reason}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    with cols[0]:
        st.markdown(_metric("Face", _pct(face)), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(_metric("Voice", _pct(voice)), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(_metric("Fusion" + (" (smoothed)" if smoothed else ""), _pct(fusion)), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(_metric("Faces used", f"{ev.faces_used}/{ev.frames_checked}"), unsafe_allow_html=True)
    with cols[4]:
        st.markdown(_metric("Speech", f"{ev.audio_seconds:.1f}s"), unsafe_allow_html=True)

    counts = Counter(ev.notes)
    for message, n in counts.items():
        st.caption(f"{message} ({n}×)" if n > 1 else message)


def render_expression(expr) -> None:
    st.markdown(
        '<div class="expr-section-title">🙂 Facial expression '
        '(advisory, does not change the decision)</div>',
        unsafe_allow_html=True,
    )
    if not expr:
        st.caption("No expression result for this input.")
        return

    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            '<div class="expr-card"><div class="expr-model-name">FERPlus</div>',
            unsafe_allow_html=True,
        )
        if expr.get("fer_top"):
            top = expr["fer_top"]
            prob = expr["fer_probs"].get(top, 0.0)
            st.markdown(
                f'<div class="expr-primary-label">{top.title()}'
                f'<span class="expr-confidence"> — {prob:.1%}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Unavailable")
        st.markdown("</div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(
            '<div class="expr-card"><div class="expr-model-name">MobileFaceNet</div>',
            unsafe_allow_html=True,
        )
        if expr.get("mobile_top"):
            st.markdown(
                f'<div class="expr-primary-label">{expr["mobile_top"].title()}</div>',
                unsafe_allow_html=True,
            )
            st.caption("Label only, no calibrated probability")
        else:
            st.caption("Unavailable")
        st.markdown("</div>", unsafe_allow_html=True)

    if expr.get("agree") is True:
        st.markdown('<div class="expr-agreement agree">✓ The two expression models agree</div>', unsafe_allow_html=True)
    elif expr.get("agree") is False:
        st.markdown('<div class="expr-agreement disagree">✗ The two expression models disagree</div>', unsafe_allow_html=True)
    st.caption("Expression is not the same as stress. The stress decision uses only the face and voice models.")


def render_model_details() -> None:
    with st.expander("Model details"):
        st.write(
            f"Fusion: {CFG.face_weight:.2f} face + {CFG.voice_weight:.2f} voice, "
            f"threshold {CFG.threshold:.2f}, inconclusive band ±{CFG.band:.2f}"
        )
        st.write(
            f"Feature sizes: face {CFG.face_dim}, voice {CFG.voice_dim}. "
            f"Face frames per clip: {CFG.n_face_frames}. Device: {bundle.device}"
        )
        st.write(f"Minimum face size: {CFG.min_face_px}px, minimum detection confidence: {CFG.min_face_score:.2f}")
        if bundle.expression_notes:
            for note in bundle.expression_notes:
                st.caption(note)
        else:
            st.caption("Expression models loaded.")


def render_result_block(ev: Evidence, smoothed_face=None, smoothed_voice=None, smoothed=False) -> None:
    face = smoothed_face if smoothed else ev.face_prob
    voice = smoothed_voice if smoothed else ev.voice_prob
    fusion = fuse(face, voice, CFG)
    decision = make_decision(face, voice, live_mode=smoothed)
    render_decision(decision, face, voice, fusion, ev, smoothed)
    render_expression(ev.expression)


# =============================================================================
# Mode selection
# =============================================================================

mode = st.radio(
    "Select analysis mode",
    ["📁 Upload Video", "🎥 Live Camera + Microphone"],
    horizontal=True,
)


# =============================================================================
# Upload mode
# =============================================================================

if mode == "📁 Upload Video":
    st.markdown('<div class="section-lab">Video Analysis</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a video containing a face and speech",
        type=["mp4", "mov", "avi", "mkv", "flv"],
        help="Use a clear, front-facing face and audible speech of at least 2 seconds.",
    )

    left, right = st.columns([0.92, 1.08], gap="large")

    with left:
        st.markdown('<div class="panel-kicker">01 · VIDEO PREVIEW</div>', unsafe_allow_html=True)
        if uploaded is not None:
            st.video(uploaded)
            if st.button("Analyze video", type="primary", use_container_width=True):
                with st.spinner("Analyzing face frames and speech..."):
                    path = None
                    try:
                        suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded.getvalue())
                            path = tmp.name
                        ev = inf.analyze_video(bundle, path)
                        st.session_state["upload_result"] = {
                            "signature": (uploaded.name, uploaded.size),
                            "evidence": ev,
                        }
                    except Exception as exc:
                        st.error("Video analysis failed.")
                        st.exception(exc)
                    finally:
                        if path and os.path.exists(path):
                            os.remove(path)
        else:
            st.markdown(
                '<div class="preview-empty"><div class="preview-icon">▶</div>'
                '<strong>Your video preview</strong><span>Upload a video above to begin.</span></div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="panel-kicker">02 · ANALYSIS RESULTS</div>', unsafe_allow_html=True)
        saved = st.session_state.get("upload_result")
        if saved and uploaded is not None and saved["signature"] == (uploaded.name, uploaded.size):
            render_result_block(saved["evidence"])
        else:
            st.markdown(
                '<div class="results-empty"><div class="preview-icon">✳</div>'
                '<strong>Results will appear here</strong>'
                '<span>Upload a video, then select <b>Analyze video</b>.</span></div>',
                unsafe_allow_html=True,
            )

    render_model_details()


# =============================================================================
# Live mode
# =============================================================================

else:
    st.markdown('<div class="section-lab">Live Camera &amp; Microphone</div>', unsafe_allow_html=True)

    left, right = st.columns([0.92, 1.08], gap="large")

    with left:
        st.markdown('<div class="panel-kicker">01 · LIVE PREVIEW</div>', unsafe_allow_html=True)
        st.info(
            "Allow camera and microphone access. Keep your face in frame and speak "
            "for at least 2 seconds in each window."
        )
        st.markdown(
            '<div class="live-upload-tip">Upload a video for better accuracy.</div>',
            unsafe_allow_html=True,
        )
        webrtc_ctx = webrtc_streamer(
            key="stress-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: VideoProcessor(live_state),
            audio_processor_factory=lambda: AudioProcessor(live_state),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280, "min": 640},
                    "height": {"ideal": 720, "min": 360},
                    "frameRate": {"ideal": 30, "max": 30},
                    "facingMode": {"ideal": "user"},
                },
                # Browser audio processing changes the spectrum the model was
                # trained on, so it is turned off here. Test this choice.
                "audio": {
                    "echoCancellation": False,
                    "noiseSuppression": False,
                    "autoGainControl": False,
                },
            },
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )

        is_playing = bool(webrtc_ctx.state.playing)
        was_playing = st.session_state.get("live_was_playing", False)
        if is_playing and not was_playing:
            live_state.reset()
        st.session_state.live_was_playing = is_playing

    with right:
        st.markdown('<div class="panel-kicker">02 · LIVE ANALYSIS</div>', unsafe_allow_html=True)

        if is_playing:

            @st.fragment(run_every=LIVE_REFRESH)
            def live_panel():
                frames, audio, sr = live_state.snapshot()
                if not frames:
                    st.info("Waiting for camera frames...")
                    return

                ev = inf.analyze_live(bundle, frames, audio, sr)

                alpha = CFG.ema_alpha
                if ev.face_prob is not None:
                    live_state.face = ema(live_state.face, ev.face_prob, alpha)
                if ev.voice_prob is not None:
                    live_state.voice = ema(live_state.voice, ev.voice_prob, alpha)

                render_result_block(
                    ev,
                    smoothed_face=live_state.face,
                    smoothed_voice=live_state.voice,
                    smoothed=True,
                )
                st.caption(
                    f"Refreshes every 2 seconds. Scores are smoothed over recent windows "
                    f"(alpha {alpha:.2f})."
                )

            live_panel()
        else:
            st.markdown(
                '<div class="results-empty"><div class="preview-icon">◉</div>'
                '<strong>Live results will appear here</strong>'
                '<span>Start the camera stream to begin analysis.</span></div>',
                unsafe_allow_html=True,
            )

    render_model_details()


# =============================================================================
# How it works
# =============================================================================

st.divider()
st.markdown('<div class="section-lab">How It Works</div>', unsafe_allow_html=True)

_steps = [
    ("01", "Capture", "🎬", "Upload a clip or use the camera and microphone."),
    ("02", "Face", "👤", "Faces are detected, cropped with a 15% margin and encoded by ResNet-18 into 512 values. Low-quality frames are skipped."),
    ("03", "Voice", "🎙️", "Speech is converted into 240 MFCC statistics. At least 2 seconds of speech are required."),
    ("04", "Decide", "⚖️", "Face and voice probabilities are averaged 40/60. Scores near the threshold are reported as inconclusive. Expression models are shown separately and never change the result."),
]
_cols = st.columns(4)
for col, (num, title, icon, text) in zip(_cols, _steps):
    with col:
        st.markdown(
            f"""
            <div style="padding: 16px 14px; border: 1px solid var(--border-subtle);
                        border-top: 3px solid var(--accent-main); border-radius: var(--radius-md);
                        background: var(--bg-surface); height: 100%;">
              <div style="font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
                          color: var(--accent-warm); margin-bottom: 6px;">{num}</div>
              <div style="font-size: 18px; margin-bottom: 4px;">{icon}</div>
              <div style="font-size: 14px; font-weight: 700; color: var(--text-primary);
                          margin-bottom: 8px;">{title}</div>
              <div style="font-size: 13px; color: var(--text-secondary); line-height: 1.6;">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.caption(
    "Face and voice fusion drives every decision. Expression models are advisory. "
    "Labels are emotion-derived proxies, not clinical stress diagnoses."
)

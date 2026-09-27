import os
import tempfile
import threading
from collections import deque

import av
import cv2
import joblib
import librosa
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as transforms

from PIL import Image
from sklearn.preprocessing import StandardScaler
from torchvision.models import resnet18
from streamlit_webrtc import webrtc_streamer, WebRtcMode


# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "AI Human Stress Detection"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

YU_NET_PATH = os.path.join(
    MODEL_DIR, "face_detection_yunet_2026may.onnx"
)

RESNET_PATH = os.path.join(
    MODEL_DIR, "resnet18_face_encoder.pth"
)

CLASSIFIER_PATH = os.path.join(
    MODEL_DIR, "stress_detection_model.pkl"
)

FUSION_CONFIG_PATH = os.path.join(
    MODEL_DIR, "fusion_config.json"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FACE_WEIGHT = 0.40
VOICE_WEIGHT = 0.60
THRESHOLD = 0.50

AUDIO_SR = 16000
AUDIO_WINDOW_SECONDS = 3.0

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    .hero {
        padding: 25px 30px;
        border-radius: 18px;
        background: linear-gradient(135deg, #111827, #1f2937);
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        margin-bottom: 5px;
        font-size: 34px;
    }

    .hero p {
        margin: 0;
        color: #d1d5db;
        font-size: 16px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 15px;
        background: white;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .metric-title {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 27px;
        font-weight: 700;
        color: #111827;
    }

    .result-stress {
        padding: 22px;
        border-radius: 16px;
        background: #fff1f2;
        border: 1px solid #fecdd3;
        text-align: center;
    }

    .result-safe {
        padding: 22px;
        border-radius: 16px;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        text-align: center;
    }

    .result-title {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .result-subtitle {
        color: #6b7280;
        font-size: 14px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 12px;
        color: #111827;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🧠 AI Human Stress Detection</h1>
        <p>
            Multimodal facial + voice analysis using deep learning and late fusion
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_yunet():

    if not os.path.exists(YU_NET_PATH):
        raise FileNotFoundError(
            f"YuNet model not found:\n{YU_NET_PATH}"
        )

    detector = cv2.FaceDetectorYN.create(
        YU_NET_PATH,
        "",
        (320, 320),
        0.7,
        0.3,
        5000,
    )

    return detector


@st.cache_resource
def load_face_encoder():

    if not os.path.exists(RESNET_PATH):
        raise FileNotFoundError(
            f"ResNet18 encoder not found:\n{RESNET_PATH}"
        )

    model = resnet18(weights=None)

    model.fc = torch.nn.Identity()

    checkpoint = torch.load(
        RESNET_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


@st.cache_resource
def load_classifier():

    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(
            f"Classifier file not found:\n{CLASSIFIER_PATH}"
        )

    return joblib.load(CLASSIFIER_PATH)


@st.cache_resource
def load_preprocessing():

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


try:

    yunet = load_yunet()
    face_encoder = load_face_encoder()
    classifier_package = load_classifier()
    face_transform = load_preprocessing()

except Exception as e:

    st.error("Model loading failed.")
    st.code(str(e))
    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ System")

    st.write(f"**Device:** `{DEVICE.upper()}`")

    st.divider()

    st.markdown("### Model Pipeline")

    st.write("🎥 YuNet Face Detection")
    st.write("🧠 ResNet18 Face Encoder")
    st.write("🎙️ MFCC Voice Features")
    st.write("🔗 Late Fusion")

    st.divider()

    st.markdown("### Fusion")

    st.write("Face weight: **40%**")
    st.write("Voice weight: **60%**")
    st.write("Threshold: **0.50**")


# ============================================================
# FACE DETECTION
# ============================================================

def detect_face(frame):

    """
    Detect the highest-confidence face.
    Returns cropped BGR face or None.
    """

    if frame is None:
        return None

    h, w = frame.shape[:2]

    yunet.setInputSize((w, h))

    _, faces = yunet.detect(frame)

    if faces is None or len(faces) == 0:
        return None

    best_face = max(
        faces,
        key=lambda x: float(x[14])
    )

    x, y, bw, bh = best_face[:4]

    x = int(max(0, x))
    y = int(max(0, y))
    bw = int(bw)
    bh = int(bh)

    # 15% margin
    margin_x = int(0.15 * bw)
    margin_y = int(0.15 * bh)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)

    x2 = min(w, x + bw + margin_x)
    y2 = min(h, y + bh + margin_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


# ============================================================
# FACE FEATURE EXTRACTION
# ============================================================

@torch.inference_mode()
def face_embedding(face_bgr):

    if face_bgr is None:
        return None

    face_rgb = cv2.cvtColor(
        face_bgr,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(face_rgb)

    tensor = face_transform(image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    embedding = face_encoder(tensor)

    embedding = embedding.squeeze(0).cpu().numpy()

    return embedding.astype(np.float32)


def extract_face_feature(frame):

    face = detect_face(frame)

    if face is None:
        return None

    return face_embedding(face)


# ============================================================
# VOICE FEATURES
# ============================================================

def extract_voice_features(audio, sample_rate):

    if audio is None:
        return None

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    if len(audio) < 1000:
        return None

    # Resample to project sampling rate
    if sample_rate != AUDIO_SR:

        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=AUDIO_SR,
        )

    # Normalize
    max_value = np.max(np.abs(audio))

    if max_value > 1e-8:
        audio = audio / max_value

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=AUDIO_SR,
        n_mfcc=40,
    )

    delta = librosa.feature.delta(mfcc)

    delta2 = librosa.feature.delta(
        mfcc,
        order=2,
    )

    features = np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),

            delta.mean(axis=1),
            delta.std(axis=1),

            delta2.mean(axis=1),
            delta2.std(axis=1),
        ]
    )

    return features.astype(np.float32)


# ============================================================
# MODEL PREDICTION
# ============================================================

def get_face_model():

    return classifier_package["face"]


def get_voice_model():

    return classifier_package["voice"]


def predict_face(feature):

    if feature is None:
        return None

    face_bundle = get_face_model()

    scaler = face_bundle["scaler"]
    model = face_bundle["model"]

    x = scaler.transform(
        feature.reshape(1, -1)
    )

    probability = model.predict_proba(x)[0, 1]

    return float(probability)


def predict_voice(feature):

    if feature is None:
        return None

    voice_bundle = get_voice_model()

    scaler = voice_bundle["scaler"]
    model = voice_bundle["model"]

    x = scaler.transform(
        feature.reshape(1, -1)
    )

    probability = model.predict_proba(x)[0, 1]

    return float(probability)


def fuse_predictions(
    face_probability,
    voice_probability,
):

    if face_probability is None and voice_probability is None:
        return None

    if face_probability is None:
        return voice_probability

    if voice_probability is None:
        return face_probability

    return (
        FACE_WEIGHT * face_probability
        +
        VOICE_WEIGHT * voice_probability
    )


def final_prediction(probability):

    if probability is None:
        return None

    return probability >= THRESHOLD


# ============================================================
# RESULT DISPLAY
# ============================================================

def show_result(
    fusion_probability,
    face_probability=None,
    voice_probability=None,
):

    if fusion_probability is None:

        st.warning(
            "Unable to generate a prediction. "
            "Make sure a face and/or sufficient audio are available."
        )

        return

    prediction = final_prediction(
        fusion_probability
    )

    if prediction:

        st.markdown(
            f"""
            <div class="result-stress">
                <div class="result-title">⚠️ STRESS DETECTED</div>
                <div class="result-subtitle">
                    Fusion probability: {fusion_probability:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
            <div class="result-safe">
                <div class="result-title">✓ NO STRESS DETECTED</div>
                <div class="result-subtitle">
                    Fusion probability: {fusion_probability:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:

        value = (
            f"{face_probability:.1%}"
            if face_probability is not None
            else "N/A"
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Face Probability</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        value = (
            f"{voice_probability:.1%}"
            if voice_probability is not None
            else "N/A"
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Voice Probability</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Fusion Probability</div>
                <div class="metric-value">{fusion_probability:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# VIDEO UPLOAD MODE
# ============================================================

def analyze_uploaded_video(video_bytes):

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp:

            temp.write(video_bytes)
            temp_path = temp.name

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            return None, None, None, None

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 25

        duration = (
            total_frames / fps
            if total_frames > 0
            else 0
        )

        # Sample 5 frames
        frame_indices = np.linspace(
            0,
            max(total_frames - 1, 0),
            5,
            dtype=int,
        )

        embeddings = []

        for index in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(index)
            )

            success, frame = cap.read()

            if not success:
                continue

            feature = extract_face_feature(frame)

            if feature is not None:
                embeddings.append(feature)

        cap.release()

        face_feature = None

        if embeddings:

            face_feature = np.mean(
                np.stack(embeddings),
                axis=0
            )

        # Extract audio
        audio, sr = librosa.load(
            temp_path,
            sr=AUDIO_SR,
            mono=True,
        )

        voice_feature = extract_voice_features(
            audio,
            sr,
        )

        face_probability = predict_face(
            face_feature
        )

        voice_probability = predict_voice(
            voice_feature
        )

        fusion_probability = fuse_predictions(
            face_probability,
            voice_probability,
        )

        return (
            face_probability,
            voice_probability,
            fusion_probability,
            duration,
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# LIVE PROCESSING STATE
# ============================================================

class LiveState:

    def __init__(self):

        self.lock = threading.Lock()

        self.latest_frame = None

        self.audio_chunks = deque(
            maxlen=200
        )

        self.sample_rate = AUDIO_SR

        self.running = False


if "live_state" not in st.session_state:

    st.session_state.live_state = LiveState()


live_state = st.session_state.live_state


# ============================================================
# WEBRTC PROCESSORS
# ============================================================

class VideoProcessor:

    def __init__(self, state):

        self.state = state

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        with self.state.lock:

            self.state.latest_frame = image.copy()

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


class AudioProcessor:

    def __init__(self, state):

        self.state = state

    def recv(self, frame):

        audio = frame.to_ndarray(
            format="s16"
        )

        audio = audio.astype(
            np.float32
        ) / 32768.0

        sample_rate = frame.sample_rate

        with self.state.lock:

            self.state.sample_rate = sample_rate

            self.state.audio_chunks.append(
                audio.copy()
            )

        return frame


# ============================================================
# LIVE SNAPSHOT
# ============================================================

def get_live_snapshot():

    with live_state.lock:

        frame = (
            live_state.latest_frame.copy()
            if live_state.latest_frame is not None
            else None
        )

        audio_chunks = list(
            live_state.audio_chunks
        )

        sample_rate = live_state.sample_rate

    if audio_chunks:

        audio = np.concatenate(
            [
                np.asarray(x).reshape(-1)
                for x in audio_chunks
            ]
        )

        max_samples = int(
            AUDIO_WINDOW_SECONDS * sample_rate
        )

        audio = audio[-max_samples:]

    else:

        audio = None

    return frame, audio, sample_rate


# ============================================================
# LIVE PREDICTION
# ============================================================

def analyze_live():

    frame, audio, sample_rate = get_live_snapshot()

    if frame is None:
        return None, None, None

    face_feature = extract_face_feature(
        frame
    )

    voice_feature = None

    if audio is not None:

        voice_feature = extract_voice_features(
            audio,
            sample_rate,
        )

    face_probability = predict_face(
        face_feature
    )

    voice_probability = predict_voice(
        voice_feature
    )

    fusion_probability = fuse_predictions(
        face_probability,
        voice_probability,
    )

    return (
        face_probability,
        voice_probability,
        fusion_probability,
    )


# ============================================================
# MODE SELECTION
# ============================================================

mode = st.radio(
    "Select analysis mode",
    [
        "📁 Upload Video",
        "🎥 Live Camera + Microphone",
    ],
    horizontal=True,
)


# ============================================================
# UPLOAD MODE
# ============================================================

if mode == "📁 Upload Video":

    st.markdown(
        '<div class="section-title">Analyze a Video</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a video containing a face and speech",
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv",
            "flv",
        ],
    )

    if uploaded_file is not None:

        st.video(
            uploaded_file
        )

        if st.button(
            "🔍 Analyze Video",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Analyzing facial and voice signals..."
            ):

                try:

                    (
                        face_probability,
                        voice_probability,
                        fusion_probability,
                        duration,
                    ) = analyze_uploaded_video(
                        uploaded_file.getvalue()
                    )

                    st.success(
                        f"Analysis completed • "
                        f"Video duration: {duration:.1f}s"
                    )

                    show_result(
                        fusion_probability,
                        face_probability,
                        voice_probability,
                    )

                except Exception as e:

                    st.error(
                        "Video analysis failed."
                    )

                    st.exception(e)


# ============================================================
# LIVE MODE
# ============================================================

else:

    st.markdown(
        '<div class="section-title">Live Stress Analysis</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Allow camera and microphone access. "
        "The system continuously analyzes the latest video frame "
        "and a rolling audio window."
    )

    webrtc_ctx = webrtc_streamer(
        key="stress-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: VideoProcessor(
            live_state
        ),
        audio_processor_factory=lambda: AudioProcessor(
            live_state
        ),
        media_stream_constraints={
            "video": True,
            "audio": True,
        },
        async_processing=True,
    )

    # Live prediction refresh
    if webrtc_ctx.state.playing:

        @st.fragment(run_every="2s")
        def live_dashboard():

            try:

                (
                    face_probability,
                    voice_probability,
                    fusion_probability,
                ) = analyze_live()

                if fusion_probability is None:

                    st.warning(
                        "Waiting for camera/audio data..."
                    )

                    return

                show_result(
                    fusion_probability,
                    face_probability,
                    voice_probability,
                )

                st.caption(
                    "Live prediction updates every ~2 seconds "
                    "using the latest video frame and recent audio."
                )

            except Exception as e:

                st.error(
                    "Live analysis error."
                )

                st.exception(e)

        live_dashboard()

    else:

        st.markdown(
            """
            <div class="result-safe">
                <div class="result-title">
                    🎥 Camera & Microphone Ready
                </div>
                <div class="result-subtitle">
                    Start the stream to begin live analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Facial + Voice late-fusion stress classification"
)
import os
import tempfile
import threading
from collections import deque

import av
import cv2
import joblib
import librosa
import numpy as np
import onnxruntime as ort
import streamlit as st
import torch
import torchvision.transforms as transforms

from PIL import Image
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

# Local open-source facial-expression verification model.
# Advisory only: this branch NEVER changes the stress prediction.
FER_MODEL_PATH = os.path.join(
    MODEL_DIR, "fer", "emotion-ferplus-8.onnx"
)

# Emotion FERPlus output order documented by the ONNX model:
# neutral, happiness, surprise, sadness, anger, disgust, fear, contempt.
FER_LABELS = [
    "neutral",
    "happy",
    "surprise",
    "sad",
    "angry",
    "disgust",
    "fear",
    "contempt",
]

# Project-grounded local retrieval knowledge base.
# This is deterministic retrieval, not an external LLM call.
RAG_DOCUMENTS = [
    "The stress classifier uses facial ResNet18 features and voice MFCC features.",
    "The final stress prediction uses late fusion with 40 percent face probability and 60 percent voice probability.",
    "The stress decision threshold is 0.50. This project uses emotion-derived proxy labels, not clinically validated stress labels.",
    "Facial-expression recognition identifies visible expressions such as happy, sad, angry, fear, disgust, surprise, neutral and contempt. Expression is not the same thing as a person's actual emotional or stress state.",
    "A disagreement between the stress classifier and facial-expression verifier should be reported as a modality or label disagreement, not used to overwrite the stress prediction.",
    "Happy or neutral facial expression can coexist with stress signals from voice or other modalities. Angry, fearful or disgust expressions can also occur without actual stress.",
    "Sad and surprise are treated as ambiguous for binary stress consistency checking because they do not provide a reliable binary stress decision.",
]

def _tokenize(text):
    return [
        token.strip(".,:;!?()[]{}").lower()
        for token in text.split()
        if token.strip(".,:;!?()[]{}")
    ]

RAG_DOCUMENT_TOKENS = [_tokenize(doc) for doc in RAG_DOCUMENTS]


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

# ---- Load external UI assets ----
UI_DIR = os.path.join(BASE_DIR, "ui")

def _read_ui_file(filename):
    path = os.path.join(UI_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# 1. Inject Stylesheet
st.markdown(
    f"<style>\n{_read_ui_file('styles.css')}\n</style>",
    unsafe_allow_html=True,
)

# 2. Inject 3D WebGL Background directly into the page DOM
_bg_html = _read_ui_file("background.html")
st.html(_bg_html, unsafe_allow_javascript=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-lab">
        <span class="hero-tag">Multimodal AI</span>
        <span class="hero-tag">Late Fusion</span>
        <h1>Human Stress Detection</h1>
        <p>
            Facial &amp; voice signal analysis using deep learning —
            ResNet-18 encoder · MFCC extraction · probabilistic fusion
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
def load_fer_model():
    """Load the local Emotion FERPlus ONNX model once."""
    if not os.path.exists(FER_MODEL_PATH):
        raise FileNotFoundError(
            "Facial-expression model not found:\n"
            f"{FER_MODEL_PATH}\n\n"
            "Place emotion-ferplus-8.onnx in models/fer/."
        )

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        FER_MODEL_PATH,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if not inputs:
        raise RuntimeError("FER ONNX model has no input tensor.")
    if not outputs:
        raise RuntimeError("FER ONNX model has no output tensor.")

    input_shape = inputs[0].shape
    if len(input_shape) != 4:
        raise RuntimeError(
            f"Unexpected FER input shape: {input_shape}. "
            "Expected N x 1 x 64 x 64."
        )

    return session


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
    fer_session = load_fer_model()

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
    st.write("🙂 FERPlus Expression Verification")

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
# FACIAL-EXPRESSION VERIFICATION + RAG
# ============================================================

def _softmax(scores):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)


def verify_facial_expression(face_bgr):
    """
    Predict visible facial expression using local Emotion FERPlus ONNX.

    This branch is advisory only. It never changes the trained stress
    probabilities or the 40/60 fusion decision.
    """
    if face_bgr is None or face_bgr.size == 0:
        return None

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(
        gray,
        (64, 64),
        interpolation=cv2.INTER_AREA,
    )

    # FERPlus expects N x 1 x 64 x 64.
    input_tensor = gray.astype(np.float32)[None, None, :, :]
    input_name = fer_session.get_inputs()[0].name

    try:
        outputs = fer_session.run(
            None,
            {input_name: input_tensor},
        )
    except Exception as exc:
        st.warning(
            "Facial-expression verification unavailable. "
            f"Stress prediction is unchanged. ({exc})"
        )
        return None

    if not outputs:
        return None

    probabilities = _softmax(outputs[0])

    if probabilities.shape[0] != len(FER_LABELS):
        raise RuntimeError(
            "FER model output does not contain 8 emotion scores. "
            f"Received shape: {np.asarray(outputs[0]).shape}"
        )

    emotion_probabilities = {
        label: float(probabilities[index])
        for index, label in enumerate(FER_LABELS)
    }

    top_emotion = max(
        emotion_probabilities,
        key=emotion_probabilities.get,
    )

    return {
        "top_emotion": top_emotion,
        "probabilities": emotion_probabilities,
    }


def retrieve_rag_context(query, top_k=3):
    """Retrieve the most relevant project-grounded facts locally."""
    query_tokens = set(_tokenize(query))

    if not query_tokens:
        return []

    scored = []

    for index, tokens in enumerate(RAG_DOCUMENT_TOKENS):
        overlap = len(query_tokens.intersection(tokens))

        if overlap:
            scored.append((overlap, index))

    scored.sort(key=lambda item: (-item[0], item[1]))

    return [
        RAG_DOCUMENTS[index]
        for _, index in scored[:top_k]
    ]


def build_grounded_explanation(
    fusion_probability,
    expression_result=None,
):
    stress_detected = fusion_probability >= THRESHOLD

    emotion = (
        expression_result["top_emotion"]
        if expression_result
        else "unknown"
    )

    query = (
        f"stress probability {fusion_probability:.2f} "
        f"facial expression {emotion} "
        "modality disagreement emotion derived stress proxy"
    )

    context = retrieve_rag_context(query)

    if expression_result is None:
        explanation = (
            "The stress result is based on the available multimodal signals. "
            "No facial-expression verification was available."
        )

    elif stress_detected and emotion in {"happy", "neutral"}:
        explanation = (
            f"The fusion model detects stress ({fusion_probability:.1%}), "
            f"while the expression verifier detects {emotion}. "
            "This is a potential modality disagreement. The expression "
            "verifier does not override the stress model."
        )

    elif (not stress_detected) and emotion in {
        "angry",
        "fear",
        "disgust",
    }:
        explanation = (
            f"The fusion model does not detect stress ({fusion_probability:.1%}), "
            f"while the expression verifier detects {emotion}. "
            "This is a potential modality disagreement. Visible expression "
            "alone is not sufficient to infer stress."
        )

    elif emotion in {"sad", "surprise", "contempt"}:
        explanation = (
            f"The fusion model gives {fusion_probability:.1%} stress "
            f"probability and the expression verifier detects {emotion}. "
            "This expression is treated as ambiguous for binary stress "
            "consistency checking."
        )

    else:
        explanation = (
            f"The fusion model gives {fusion_probability:.1%} stress "
            f"probability and the expression verifier detects {emotion}. "
            "The two signals are broadly consistent, but expression "
            "recognition is not a direct measurement of stress."
        )

    return explanation, context


def show_expression_verification(expression_result):
    if expression_result is None:
        return

    st.markdown("### Facial Expression Verification")

    st.write(
        f"**Detected expression:** "
        f"`{expression_result['top_emotion'].title()}`"
    )

    probabilities = sorted(
        expression_result["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True,
    )

    cols = st.columns(min(4, max(1, len(probabilities))))

    for index, (label, score) in enumerate(probabilities):
        with cols[index % len(cols)]:
            st.metric(label.title(), f"{score:.1%}")


# ============================================================
# RESULT DISPLAY
# ============================================================

def show_result(
    fusion_probability,
    face_probability=None,
    voice_probability=None,
    expression_result=None,
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
            <div class="result-lab stress">
                <div class="result-heading">⚠ STRESS DETECTED</div>
                <div class="result-detail">
                    Fusion score: {fusion_probability:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
            <div class="result-lab safe">
                <div class="result-heading">✓ No Stress Detected</div>
                <div class="result-detail">
                    Fusion score: {fusion_probability:.1%}
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
            else "—"
        )

        st.markdown(
            f"""
            <div class="metric-lab">
                <div class="metric-label">Face</div>
                <div class="metric-val">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        value = (
            f"{voice_probability:.1%}"
            if voice_probability is not None
            else "—"
        )

        st.markdown(
            f"""
            <div class="metric-lab">
                <div class="metric-label">Voice</div>
                <div class="metric-val">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-lab">
                <div class="metric-label">Fusion</div>
                <div class="metric-val">{fusion_probability:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    show_expression_verification(expression_result)

    if fusion_probability is not None:
        explanation, rag_context = build_grounded_explanation(
            fusion_probability,
            expression_result,
        )

        with st.expander("🧠 Grounded RAG Explanation", expanded=False):
            st.write(explanation)

            if rag_context:
                st.markdown("**Retrieved project knowledge:**")
                for item in rag_context:
                    st.caption(f"• {item}")


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
        expression_predictions = []

        for index in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(index)
            )

            success, frame = cap.read()

            if not success:
                continue

            face = detect_face(frame)

            if face is not None:
                feature = face_embedding(face)

                if feature is not None:
                    embeddings.append(feature)

                expression = verify_facial_expression(face)

                if expression is not None:
                    expression_predictions.append(expression)

        cap.release()

        face_feature = None

        if embeddings:

            face_feature = np.mean(
                np.stack(embeddings),
                axis=0
            )

        expression_result = None

        if expression_predictions:
            labels = set()

            for item in expression_predictions:
                labels.update(item["probabilities"].keys())

            averaged = {
                label: float(
                    np.mean(
                        [
                            item["probabilities"].get(label, 0.0)
                            for item in expression_predictions
                        ]
                    )
                )
                for label in labels
            }

            expression_result = {
                "top_emotion": max(averaged, key=averaged.get),
                "probabilities": averaged,
            }

        # Extract audio
        def decode_audio_from_video(video_path):

            with av.open(video_path) as container:

                audio_streams = container.streams.audio

                if not audio_streams:
                    raise ValueError(
                        "The uploaded video does not contain an audio stream."
                    )

                audio_stream = audio_streams[0]

                resampler = av.audio.resampler.AudioResampler(
                    format="flt",
                    layout="mono",
                    rate=AUDIO_SR,
                )

                chunks = []

                def append_chunk(audio_frame):

                    chunk = audio_frame.to_ndarray()
                    chunk = np.asarray(chunk, dtype=np.float32)

                    if chunk.ndim == 2:
                        if chunk.shape[0] == 1:
                            chunk = chunk[0]
                        else:
                            chunk = np.mean(chunk, axis=0)
                    else:
                        chunk = chunk.reshape(-1)

                    if chunk.size > 0:
                        chunks.append(chunk)

                for packet in container.demux(audio_stream):

                    for frame in packet.decode():

                        for resampled in resampler.resample(frame):
                            append_chunk(resampled)

                for resampled in resampler.resample(None):
                    append_chunk(resampled)

                if not chunks:
                    raise ValueError(
                        "Could not decode audio from the uploaded video."
                    )

                return np.concatenate(chunks).astype(
                    np.float32,
                    copy=False,
                )

        audio = decode_audio_from_video(temp_path)

        voice_feature = extract_voice_features(
            audio,
            AUDIO_SR,
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
            expression_result,
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

    face = detect_face(frame)

    face_feature = (
        face_embedding(face)
        if face is not None
        else None
    )

    expression_result = (
        verify_facial_expression(face)
        if face is not None
        else None
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
        expression_result,
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
        '<div class="section-lab">Analyze a Video</div>',
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
                        expression_result,
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
                        expression_result,
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
        '<div class="section-lab">Live Stress Analysis</div>',
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
                    expression_result,
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
                    expression_result,
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
            <div class="result-lab safe">
                <div class="result-heading">
                    Camera &amp; Microphone Ready
                </div>
                <div class="result-detail">
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
    "Facial + Voice late-fusion stress classification • FERPlus is advisory"
)
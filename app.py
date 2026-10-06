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

# Expression consistency gate. The original app always let the trained
# stress classifier win, even when both expression models strongly agreed
# on a low-stress expression. This gate prevents that specific false-positive
# pattern without retraining or adding another model.
EXPRESSION_GATE_ENABLED = True
EXPRESSION_GATE_MIN_FER_CONFIDENCE = 0.60
EXPRESSION_GATE_MAX_FUSION = 0.90
LOW_STRESS_EXPRESSIONS = {"happy", "neutral"}

AUDIO_SR = 16000
AUDIO_WINDOW_SECONDS = 3.0

# Local open-source facial-expression verification model.
# Advisory only: this branch NEVER changes the stress prediction.
FER_MODEL_PATH = os.path.join(
    MODEL_DIR, "fer", "emotion-ferplus-8.onnx"
)

MOBILEFER_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "fer",
    "facial_expression_recognition_mobilefacenet_2022july.onnx",
)

# MobileFaceNet Progressive Teacher output order from OpenCV Zoo:
# angry, disgust, fearful, happy, neutral, sad, surprised.
MOBILEFER_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

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
    "A disagreement between the stress classifier and facial-expression verifier should be reported. When both expression models strongly agree on happy or neutral, the application uses a conservative expression-consistency gate to prevent a borderline stress false positive; this does not retrain the stress model.",
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
def load_mobileface_model():
    """Load OpenCV Zoo MobileFaceNet using the official OpenCV DNN path."""
    if not os.path.exists(MOBILEFER_MODEL_PATH):
        return None

    # The OpenCV Zoo model is an expression classifier whose documented
    # inference path uses cv.dnn.readNet() and returns a class label, not
    # a 7-element probability vector. Using ONNX Runtime here incorrectly
    # treated the label output as probabilities and caused the model to be
    # reported as unavailable.
    net = cv2.dnn.readNet(MOBILEFER_MODEL_PATH)
    if net.empty():
        raise RuntimeError(
            "MobileFaceNet ONNX model could not be loaded by OpenCV DNN."
        )

    return net


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
    mobilefer_session = load_mobileface_model()

except Exception as e:
    st.error("Model loading failed.")
    st.code(str(e))
    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

# (Sidebar removed for cleaner UI focus)


# ============================================================
# FACE DETECTION
# ============================================================

def detect_face_data(frame):
    """
    Detect the highest-confidence face.
    Returns (cropped_face_bgr, raw_yunet_detection) or None.
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

    margin_x = int(0.15 * bw)
    margin_y = int(0.15 * bh)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)

    x2 = min(w, x + bw + margin_x)
    y2 = min(h, y + bh + margin_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2], best_face


def detect_face(frame):
    """Detect and return the highest-confidence cropped face."""
    result = detect_face_data(frame)
    return result[0] if result is not None else None


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


def expression_consistency_gate(fusion_probability, expression_result):
    """
    Decide whether strong expression evidence should veto a stress false
    positive. This is a decision-layer rule, not a new ML model.

    We only veto when: 
      1. both FERPlus and MobileFaceNet are available,
      2. both agree on happy/neutral,
      3. FERPlus gives that expression >= 60%, and
      4. the multimodal stress score is below 90%.

    The 90% ceiling prevents the expression model from overriding a very
    strong multimodal stress signal.
    """
    if not EXPRESSION_GATE_ENABLED or fusion_probability is None:
        return None

    if not expression_result:
        return None

    fer = expression_result.get("ferplus") or {}
    mobile = expression_result.get("mobileface") or {}

    if fer.get("status") != "ready" or mobile.get("status") != "ready":
        return None

    fer_emotion = fer.get("top_emotion")
    mobile_emotion = mobile.get("top_emotion")

    if fer_emotion != mobile_emotion:
        return None

    if fer_emotion not in LOW_STRESS_EXPRESSIONS:
        return None

    fer_confidence = float(
        (fer.get("probabilities") or {}).get(fer_emotion, 0.0)
    )

    if fer_confidence < EXPRESSION_GATE_MIN_FER_CONFIDENCE:
        return None

    if fusion_probability >= EXPRESSION_GATE_MAX_FUSION:
        return None

    return {
        "prediction": False,
        "expression": fer_emotion,
        "fer_confidence": fer_confidence,
        "reason": (
            f"Both facial-expression models agree on {fer_emotion} "
            f"(FERPlus {fer_confidence:.1%}); the stress result was "
            "classified as a facial-expression contradiction."
        ),
    }


def get_final_decision(fusion_probability, expression_result):
    """Return (boolean decision, reason)."""
    base_prediction = final_prediction(fusion_probability)
    gate = expression_consistency_gate(
        fusion_probability,
        expression_result,
    )

    if gate is not None and base_prediction is True:
        return False, gate["reason"]

    return base_prediction, None


# ============================================================
# FACIAL-EXPRESSION VERIFICATION + RAG
# ============================================================

def _softmax(scores):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)


def _average_emotion_probabilities(results, labels):
    """Average probability dictionaries from multiple expression predictions."""
    if not results:
        return None

    averaged = {}
    for label in labels:
        values = [
            item["probabilities"].get(label, 0.0)
            for item in results
        ]
        averaged[label] = float(np.mean(values))

    total = sum(averaged.values())
    if total > 0:
        averaged = {
            label: value / total
            for label, value in averaged.items()
        }

    return averaged


def _align_face_for_mobilefer(frame, face_row):
    """Align a YuNet 5-landmark face for the OpenCV MobileFaceNet model."""
    landmarks = np.asarray(face_row[4:14], dtype=np.float32).reshape(5, 2)

    reference = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    transform_matrix, _ = cv2.estimateAffinePartial2D(
        landmarks,
        reference,
        method=cv2.LMEDS,
    )

    if transform_matrix is None:
        return None

    return cv2.warpAffine(
        frame,
        transform_matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def verify_mobileface_expression(frame, face_row):
    """
    Predict visible facial expression using the OpenCV Zoo MobileFaceNet
    Progressive Teacher model.

    OpenCV Zoo documents this model as a 7-class classifier with a label
    output. It uses a 112x112 aligned face and returns the class index.
    Advisory only: this branch never changes the trained stress prediction.
    """
    if mobilefer_session is None or frame is None or face_row is None:
        return None

    aligned = _align_face_for_mobilefer(frame, face_row)

    if aligned is None:
        return None

    try:
        # Match the official OpenCV Zoo preprocessing:
        # BGR -> RGB, /255, then (x - 0.5) / 0.5, NCHW blob.
        image = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5

        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=(112, 112),
            mean=(0.0, 0.0, 0.0),
            swapRB=False,
            crop=False,
        )

        mobilefer_session.setInput(blob, "data")
        output = mobilefer_session.forward("label")

        # OpenCV Zoo's post-processing is argmax(output[0]).
        label_index = int(np.asarray(output).reshape(-1)[0])

        if not 0 <= label_index < len(MOBILEFER_LABELS):
            return None

        top_emotion = MOBILEFER_LABELS[label_index]

        # This model exposes a class label rather than calibrated class
        # probabilities. Use a one-hot representation only for the UI and
        # ensemble bookkeeping; do not present it as model confidence.
        emotion_probabilities = {label: 0.0 for label in MOBILEFER_LABELS}
        emotion_probabilities[top_emotion] = 1.0

        return {
            "top_emotion": top_emotion,
            "probabilities": emotion_probabilities,
            "confidence_available": False,
            "status": "ready",
        }

    except Exception as exc:
        return {
            "top_emotion": None,
            "probabilities": {},
            "confidence_available": False,
            "status": "error",
            "error": str(exc),
        }


def verify_facial_expression(face_bgr, frame=None, face_row=None):
    """
    Run both local open-source facial-expression models.

    FERPlus provides an 8-class probability distribution. MobileFaceNet
    provides a 7-class expression label through the official OpenCV DNN
    interface. Both are advisory expression evidence only and never alter
    the trained stress probabilities or 40/60 stress fusion.
    """
    if face_bgr is None or face_bgr.size == 0:
        return None

    # ---------------- FERPlus ----------------
    fer_result = None
    fer_error = None

    try:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(
            gray,
            (64, 64),
            interpolation=cv2.INTER_AREA,
        )

        input_tensor = gray.astype(np.float32)[None, None, :, :]
        input_name = fer_session.get_inputs()[0].name
        outputs = fer_session.run(
            None,
            {input_name: input_tensor},
        )

        if not outputs:
            raise RuntimeError("FERPlus returned no output.")

        probabilities = _softmax(outputs[0])
        if probabilities.shape[0] != len(FER_LABELS):
            raise RuntimeError(
                f"FERPlus returned {probabilities.shape[0]} scores; "
                f"expected {len(FER_LABELS)}."
            )

        fer_probabilities = {
            label: float(probabilities[index])
            for index, label in enumerate(FER_LABELS)
        }

        fer_result = {
            "top_emotion": max(
                fer_probabilities,
                key=fer_probabilities.get,
            ),
            "probabilities": fer_probabilities,
            "confidence_available": True,
            "status": "ready",
        }

    except Exception as exc:
        fer_error = str(exc)
        fer_result = {
            "top_emotion": None,
            "probabilities": {},
            "confidence_available": False,
            "status": "error",
            "error": fer_error,
        }

    # ---------------- MobileFaceNet ----------------
    mobile_result = None
    if frame is not None and face_row is not None:
        mobile_result = verify_mobileface_expression(
            frame,
            face_row,
        )

    mobile_ready = (
        mobile_result is not None
        and mobile_result.get("status") == "ready"
        and mobile_result.get("top_emotion") is not None
    )
    fer_ready = (
        fer_result is not None
        and fer_result.get("status") == "ready"
        and fer_result.get("top_emotion") is not None
    )

    if not fer_ready and not mobile_ready:
        return {
            "top_emotion": None,
            "probabilities": {},
            "ferplus": fer_result,
            "mobileface": mobile_result,
            "status": "error",
        }

    # ---------------- Combined expression ----------------
    if fer_ready and mobile_ready:
        fer_top = fer_result["top_emotion"]
        mobile_top = mobile_result["top_emotion"]

        if fer_top == mobile_top:
            combined_emotion = fer_top
            agreement = True
        else:
            # FERPlus has a real probability distribution; use it as the
            # primary combined distribution while retaining MobileFaceNet as
            # an independent second opinion.
            combined_emotion = fer_top
            agreement = False

        combined_probabilities = dict(fer_result["probabilities"])
        return {
            "top_emotion": combined_emotion,
            "probabilities": combined_probabilities,
            "ferplus": fer_result,
            "mobileface": mobile_result,
            "models_available": ["FERPlus", "MobileFaceNet"],
            "agreement": agreement,
            "status": "ready",
        }

    if fer_ready:
        return {
            "top_emotion": fer_result["top_emotion"],
            "probabilities": dict(fer_result["probabilities"]),
            "ferplus": fer_result,
            "mobileface": mobile_result,
            "models_available": ["FERPlus"],
            "agreement": None,
            "status": "partial",
        }

    return {
        "top_emotion": mobile_result["top_emotion"],
        "probabilities": dict(mobile_result["probabilities"]),
        "ferplus": fer_result,
        "mobileface": mobile_result,
        "models_available": ["MobileFaceNet"],
        "agreement": None,
        "status": "partial",
    }


def aggregate_expression_predictions(predictions):
    """Aggregate sampled-frame expression results while preserving per-model status."""
    valid = [
        item for item in predictions
        if item and item.get("top_emotion")
    ]

    if not valid:
        return None

    labels = set()
    for item in valid:
        labels.update(item.get("probabilities", {}).keys())

    averaged = {}
    for label in labels:
        values = [
            item.get("probabilities", {}).get(label, 0.0)
            for item in valid
        ]
        averaged[label] = float(np.mean(values))

    total = sum(averaged.values())
    if total > 0:
        averaged = {
            label: value / total
            for label, value in averaged.items()
        }

    # Preserve the most recent successful result from each model so the UI
    # can show that the individual open-source models actually ran.
    latest_fer = None
    latest_mobile = None
    for item in valid:
        fer = item.get("ferplus")
        mobile = item.get("mobileface")
        if fer and fer.get("status") == "ready":
            latest_fer = fer
        if mobile and mobile.get("status") == "ready":
            latest_mobile = mobile

    result = {
        "top_emotion": max(averaged, key=averaged.get),
        "probabilities": averaged,
        "ferplus": latest_fer,
        "mobileface": latest_mobile,
        "status": "ready",
    }

    if latest_fer and latest_mobile:
        result["models_available"] = ["FERPlus", "MobileFaceNet"]
        result["agreement"] = (
            latest_fer.get("top_emotion") == latest_mobile.get("top_emotion")
        )
    elif latest_fer:
        result["models_available"] = ["FERPlus"]
        result["agreement"] = None
    elif latest_mobile:
        result["models_available"] = ["MobileFaceNet"]
        result["agreement"] = None

    return result


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

    top_emotion = expression_result.get("top_emotion")
    if top_emotion:
        st.write(f"**Combined expression:** `{top_emotion.title()}`")
    else:
        st.warning("Facial-expression verification could not be completed.")
        return

    model_cols = st.columns(2)

    with model_cols[0]:
        ferplus = expression_result.get("ferplus") or {}
        st.caption("FERPlus")
        if ferplus.get("status") == "ready":
            st.success(ferplus["top_emotion"].title())
        else:
            st.error("Error")
            if ferplus.get("error"):
                st.caption(ferplus["error"][:180])

    with model_cols[1]:
        mobileface = expression_result.get("mobileface") or {}
        st.caption("MobileFaceNet")
        if mobileface.get("status") == "ready":
            st.success(mobileface["top_emotion"].title())
        else:
            st.error("Error")
            if mobileface.get("error"):
                st.caption(mobileface["error"][:180])

    if expression_result.get("agreement") is True:
        st.caption("✓ Both expression models agree on the top expression.")
    elif expression_result.get("agreement") is False:
        st.caption("Different expression outputs; FERPlus distribution is shown as the combined distribution.")

    probabilities = sorted(
        expression_result.get("probabilities", {}).items(),
        key=lambda item: item[1],
        reverse=True,
    )

    if not probabilities:
        return

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

    prediction, decision_reason = get_final_decision(
        fusion_probability,
        expression_result,
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

    if decision_reason:
        st.info(
            f"Expression consistency check: {decision_reason}"
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

            face_data = detect_face_data(frame)

            if face_data is not None:
                face, face_row = face_data

                feature = face_embedding(face)

                if feature is not None:
                    embeddings.append(feature)

                expression = verify_facial_expression(
                    face,
                    frame=frame,
                    face_row=face_row,
                )

                if expression is not None:
                    expression_predictions.append(expression)

        cap.release()

        face_feature = None

        if embeddings:

            face_feature = np.mean(
                np.stack(embeddings),
                axis=0
            )

        expression_result = aggregate_expression_predictions(
            expression_predictions
        )

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

        # Rolling video buffer. Live inference samples several frames
        # from this buffer to match the offline multi-frame face pipeline.
        self.frame_buffer = deque(maxlen=30)

        self.audio_chunks = deque(maxlen=200)
        self.sample_rate = AUDIO_SR

        # Temporal smoothing for live probabilities.
        self.face_history = deque(maxlen=5)
        self.voice_history = deque(maxlen=5)
        self.fusion_history = deque(maxlen=5)

        self.expression_history = deque(maxlen=7)

        self.running = False

    def reset_for_new_stream(self):
        with self.lock:
            self.frame_buffer.clear()
            self.audio_chunks.clear()
            self.face_history.clear()
            self.voice_history.clear()
            self.fusion_history.clear()
            self.expression_history.clear()
            self.sample_rate = AUDIO_SR


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
            self.state.frame_buffer.append(
                image.copy()
            )

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

        frames = list(
            live_state.frame_buffer
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

    return frames, audio, sample_rate


def _temporal_average(history, value):

    if value is None:
        return None

    history.append(
        float(value)
    )

    return float(
        np.mean(
            np.asarray(history, dtype=np.float32)
        )
    )


def _average_live_expressions(history):
    """Aggregate live expression history and preserve model diagnostics."""
    if not history:
        return None

    return aggregate_expression_predictions(list(history))


# ============================================================
# LIVE PREDICTION
# ============================================================

def analyze_live():

    frames, audio, sample_rate = get_live_snapshot()

    if not frames:
        return None, None, None, None

    # Sample 5 frames across the rolling buffer, matching the offline
    # face pipeline's multi-frame mean-pooling strategy.
    sample_count = min(5, len(frames))
    sample_indices = np.linspace(
        0,
        len(frames) - 1,
        sample_count,
        dtype=int,
    )

    embeddings = []
    expression_predictions = []

    for index in sample_indices:

        frame = frames[int(index)]

        face_data = detect_face_data(frame)

        if face_data is None:
            continue

        face, face_row = face_data

        feature = face_embedding(face)

        if feature is not None:
            embeddings.append(feature)

        expression = verify_facial_expression(
            face,
            frame=frame,
            face_row=face_row,
        )

        if expression is not None:
            expression_predictions.append(expression)

    face_feature = None

    if embeddings:
        face_feature = np.mean(
            np.stack(embeddings),
            axis=0,
        )

    face_probability = predict_face(
        face_feature
    )

    voice_feature = None

    if audio is not None:
        voice_feature = extract_voice_features(
            audio,
            sample_rate,
        )

    voice_probability = predict_voice(
        voice_feature
    )

    smoothed_face = _temporal_average(
        live_state.face_history,
        face_probability,
    )

    smoothed_voice = _temporal_average(
        live_state.voice_history,
        voice_probability,
    )

    fusion_probability = fuse_predictions(
        smoothed_face,
        smoothed_voice,
    )

    smoothed_fusion = _temporal_average(
        live_state.fusion_history,
        fusion_probability,
    )

    expression_result = None

    if expression_predictions:
        for item in expression_predictions:
            live_state.expression_history.append(item)

        expression_result = _average_live_expressions(
            live_state.expression_history
        )

    return (
        smoothed_face,
        smoothed_voice,
        smoothed_fusion,
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
        "Live analysis uses multiple recent face frames and a rolling audio window."
    )

    if mobilefer_session is None:
        st.caption(
            "MobileFaceNet expression verification is unavailable; "
            "the stress model remains fully operational."
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

    # Reset rolling state whenever a new live session starts so previous
    # predictions cannot bias the next session.
    is_playing = bool(webrtc_ctx.state.playing)
    was_playing = st.session_state.get(
        "live_was_playing",
        False,
    )

    if is_playing and not was_playing:
        live_state.reset_for_new_stream()

    st.session_state.live_was_playing = is_playing

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
                    "Live prediction updates every ~2 seconds using "
                    "multi-frame facial analysis, temporal smoothing, "
                    "and recent audio."
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
# MODEL PIPELINE & HOW IT WORKS
# ============================================================

st.divider()

st.markdown('<div class="section-lab">Model Pipeline</div>', unsafe_allow_html=True)

st.html("""
<div class="pipeline-wrap">

  <div class="pipe-row">
    <div class="pipe-box pipe-input">
      <div class="pipe-icon">🎥</div>
      <div class="pipe-title">Video Input</div>
      <div class="pipe-desc">You upload a video or stream live camera footage.
      The video must contain a visible face and audible speech.</div>
    </div>
  </div>

  <div class="pipe-arrow">↓</div>

  <div class="pipe-split">

    <div class="pipe-track">
      <div class="pipe-track-label">👤 Face Track</div>

      <div class="pipe-box">
        <div class="pipe-title">YuNet — Face Detection</div>
        <div class="pipe-desc">Finds the person's face in the video frame.
        Only the face region is passed forward; the background is ignored.</div>
      </div>

      <div class="pipe-arrow">↓</div>

      <div class="pipe-box">
        <div class="pipe-title">ResNet18 — Feature Encoder</div>
        <div class="pipe-desc">A deep neural network that converts the face image
        into 512 numbers capturing subtle visual patterns linked to stress.</div>
      </div>

      <div class="pipe-arrow">↓</div>

      <div class="pipe-box pipe-model">
        <div class="pipe-title">Face Classifier (MLP)</div>
        <div class="pipe-desc">Reads those 512 numbers and outputs a stress
        probability between 0 and 1. Weight in fusion: <b>40%</b>.</div>
      </div>

      <div class="pipe-arrow">↘</div>
    </div>

    <div class="pipe-track">
      <div class="pipe-track-label">🎙️ Voice Track</div>

      <div class="pipe-box">
        <div class="pipe-title">Audio Extraction</div>
        <div class="pipe-desc">The audio channel is pulled from the video
        and resampled to 16 kHz for consistent processing.</div>
      </div>

      <div class="pipe-arrow">↓</div>

      <div class="pipe-box">
        <div class="pipe-title">MFCC — Voice Features</div>
        <div class="pipe-desc">Mel-Frequency Cepstral Coefficients capture
        speech texture — pitch, energy, tempo — producing 240 numbers
        (MFCC + delta + delta-delta, mean &amp; std).</div>
      </div>

      <div class="pipe-arrow">↓</div>

      <div class="pipe-box pipe-model">
        <div class="pipe-title">Voice Classifier (Logistic Regression)</div>
        <div class="pipe-desc">Reads those 240 numbers and outputs a stress
        probability. Weight in fusion: <b>60%</b>.</div>
      </div>

      <div class="pipe-arrow">↙</div>
    </div>
  </div>

  <div class="pipe-row">
    <div class="pipe-box pipe-fusion">
      <div class="pipe-icon">⚖️</div>
      <div class="pipe-title">Late Fusion</div>
      <div class="pipe-desc">
        The two probabilities are combined into one final score:<br>
        <code>Final = 0.40 × Face + 0.60 × Voice</code><br>
        The trained fusion configuration gives voice a 60% weight and face a 40% weight.
      </div>
    </div>
  </div>

  <div class="pipe-arrow">↓</div>

  <div class="pipe-row pipe-row-split2">
    <div class="pipe-box pipe-stress">
      <div class="pipe-icon">⚠️</div>
      <div class="pipe-title">Stress Detected</div>
      <div class="pipe-desc">Final score ≥ 0.50</div>
    </div>
    <div class="pipe-box pipe-safe">
      <div class="pipe-icon">✓</div>
      <div class="pipe-title">No Stress</div>
      <div class="pipe-desc">Final score &lt; 0.50</div>
    </div>
  </div>

  <div class="pipe-support-label">Supporting Layers — do not change the stress prediction</div>

  <div class="pipe-row pipe-row-split2">
    <div class="pipe-box pipe-support">
      <div class="pipe-icon">🙂</div>
      <div class="pipe-title">FERPlus + MobileFaceNet — Expression Verification</div>
      <div class="pipe-desc">Open-source expression models check the face for visible
      emotions (happy, sad, angry, fear, neutral …). It provides supporting
      evidence — a disagreement is reported, <em>not</em> used to override the stress score.</div>
    </div>
    <div class="pipe-box pipe-support">
      <div class="pipe-icon">📖</div>
      <div class="pipe-title">RAG — Grounded Explanation</div>
      <div class="pipe-desc">Retrieval-Augmented Generation searches a
      built-in knowledge base and writes a plain-English explanation
      of why the model made its decision. No external API is used.</div>
    </div>
  </div>

</div>
""")

# ---- How It Works ----
st.markdown('<div class="section-lab">How It Works</div>', unsafe_allow_html=True)

steps = [
    ("01", "Capture", "🎬",
     "Upload a video (or go live) showing a person speaking. Both face and audible speech are needed for the best prediction."),
    ("02", "Analyze the Face", "👤",
     "The system finds the face using YuNet, then runs it through ResNet-18 — a neural network that extracts 512 visual features tied to stress-related appearance."),
    ("03", "Analyze the Voice", "🎙️",
     "Audio is converted into MFCC features — a compact fingerprint of how speech sounds, capturing pitch, energy, and tempo, which all change under stress."),
    ("04", "Combine the Signals", "⚖️",
     "The face model and voice model each produce an independent stress probability. They are blended (40% face + 60% voice). A final score ≥ 0.50 means stress is detected."),
    ("05", "Verify & Explain", "📖",
     "FERPlus checks the visible facial expression. RAG generates a plain-English explanation. Neither changes the stress score — they help you understand the result."),
]

step_cols = st.columns(5)
for i, (num, title, icon, desc) in enumerate(steps):
    with step_cols[i]:
        st.markdown(
            f"""
            <div style="padding: 16px 14px; border: 1px solid var(--border-subtle);
                        border-top: 3px solid var(--accent-main);
                        border-radius: var(--radius-md);
                        background: var(--bg-surface);
                        height: 100%;">
              <div style="font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
                          color: var(--accent-warm); margin-bottom: 6px;">{num}</div>
              <div style="font-size: 18px; margin-bottom: 4px;">{icon}</div>
              <div style="font-size: 14px; font-weight: 700; color: var(--text-primary);
                          margin-bottom: 8px;">{title}</div>
              <div style="font-size: 13px; color: var(--text-secondary); line-height: 1.6;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Facial + Voice late-fusion stress classification • FERPlus is advisory"
)
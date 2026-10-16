"""Model loading, preprocessing and feature extraction for the stress-proxy app.

The constants here mirror the training script (major.py). Changing one of them
without retraining moves the features away from what the classifiers were fit on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import av
import cv2
import joblib
import librosa
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms.functional as TF
from torchvision.models import resnet18

from decision import (
    Config,
    audio_long_enough,
    face_quality_problems,
    load_config,
    majority_label,
    sample_positions,
)

# ---- Audio: must match major.py exactly -------------------------------------
SR = 16000
N_MFCC = 40
N_FFT = 512        # the old app used librosa's default 2048 here: a train/serve mismatch
HOP_LENGTH = 256   # the old app used librosa's default 512 here

# ---- Face: must match major.py ----------------------------------------------
FACE_MARGIN = 0.15          # square margin, as in training
FACE_INPUT = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
YUNET_INPUT = (320, 320)
YUNET_SCORE = 0.7
YUNET_NMS = 0.3
YUNET_TOPK = 5000
MAX_FRAME_WIDTH = 640       # upload and live frames share one scale

# ---- Gates that are guesses: check them against the training clips ----------
SILENCE_RMS = 0.002

# ---- Expression models (advisory only) --------------------------------------
FER_LABELS = [
    "neutral", "happy", "surprise", "sad",
    "angry", "disgust", "fear", "contempt",
]
MOBILEFER_LABELS = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
]
MOBILEFER_REFERENCE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class Bundle:
    cfg: Config
    device: str
    yunet: object
    encoder: torch.nn.Module
    face_model: object
    face_scaler: object
    face_index: int
    voice_model: object
    voice_scaler: object
    voice_index: int
    fer: Optional[object] = None
    mobilefer: Optional[object] = None
    expression_notes: List[str] = field(default_factory=list)


@dataclass
class FaceCrop:
    crop_bgr: np.ndarray
    frame_bgr: np.ndarray
    landmarks: np.ndarray
    score: float


@dataclass
class Evidence:
    face_prob: Optional[float] = None
    voice_prob: Optional[float] = None
    faces_used: int = 0
    frames_checked: int = 0
    audio_seconds: float = 0.0
    expression: Optional[Dict] = None
    notes: List[str] = field(default_factory=list)


# =============================================================================
# Loading
# =============================================================================

def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required model file missing: {path}")
    return path


def _check_dim(name: str, scaler, expected: int) -> None:
    n = getattr(scaler, "n_features_in_", None)
    if n is not None and int(n) != expected:
        raise RuntimeError(
            f"{name} scaler expects {n} features but fusion_config.json says "
            f"{expected}. The model files and config come from different runs."
        )


def _positive_index(model, name: str) -> int:
    classes = list(getattr(model, "classes_", []))
    for i, c in enumerate(classes):
        if str(c).strip().lower() in {"1", "stress", "stressed", "positive"}:
            return i
    raise RuntimeError(f"No stress class found for {name} model; classes_={classes!r}")


def _load_expression_models(models: Path) -> Tuple[Optional[object], Optional[object], List[str]]:
    notes: List[str] = []
    fer = None
    mobilefer = None

    fer_path = models / "fer" / "emotion-ferplus-8.onnx"
    try:
        _require(fer_path)
        fer = ort.InferenceSession(str(fer_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        notes.append(f"FERPlus unavailable: {exc}")

    mob_path = models / "fer" / "facial_expression_recognition_mobilefacenet_2022july.onnx"
    try:
        _require(mob_path)
        net = cv2.dnn.readNet(str(mob_path))
        if net.empty():
            raise RuntimeError("OpenCV could not read the model")
        mobilefer = net
    except Exception as exc:
        notes.append(f"MobileFaceNet unavailable: {exc}")

    return fer, mobilefer, notes


def load_bundle(base_dir) -> Bundle:
    models = Path(base_dir) / "models"
    cfg = load_config(models / "fusion_config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    yunet_path = _require(models / "face_detection_yunet_2026may.onnx")
    yunet = cv2.FaceDetectorYN.create(
        str(yunet_path), "", YUNET_INPUT, YUNET_SCORE, YUNET_NMS, YUNET_TOPK
    )

    encoder = resnet18(weights=None)
    encoder.fc = torch.nn.Identity()
    encoder.load_state_dict(
        torch.load(_require(models / "resnet18_face_encoder.pth"), map_location=device)
    )
    encoder.to(device).eval()

    pkg = joblib.load(_require(models / "stress_detection_model.pkl"))
    face = pkg["face"]
    voice = pkg["voice"]
    _check_dim("face", face["scaler"], cfg.face_dim)
    _check_dim("voice", voice["scaler"], cfg.voice_dim)

    fer, mobilefer, notes = _load_expression_models(models)

    return Bundle(
        cfg=cfg,
        device=device,
        yunet=yunet,
        encoder=encoder,
        face_model=face["model"],
        face_scaler=face["scaler"],
        face_index=_positive_index(face["model"], "face"),
        voice_model=voice["model"],
        voice_scaler=voice["scaler"],
        voice_index=_positive_index(voice["model"], "voice"),
        fer=fer,
        mobilefer=mobilefer,
        expression_notes=notes,
    )


# =============================================================================
# Frames and faces
# =============================================================================

def limit_width(frame_bgr: np.ndarray, max_width: int = MAX_FRAME_WIDTH) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w <= max_width:
        return frame_bgr
    new_h = int(round(h * max_width / w))
    return cv2.resize(frame_bgr, (max_width, new_h), interpolation=cv2.INTER_AREA)


def detect_face(b: Bundle, frame_bgr: np.ndarray) -> Tuple[Optional[FaceCrop], List[str]]:
    """Return the most confident face if it passes the quality gates, else the reasons."""
    h, w = frame_bgr.shape[:2]
    b.yunet.setInputSize((w, h))
    _, dets = b.yunet.detect(frame_bgr)
    if dets is None or len(dets) == 0:
        return None, ["no face found"]

    # YuNet layout: x, y, w, h, 10 landmark values, then the confidence score.
    row = dets[int(np.argmax(dets[:, 14]))]
    x, y, bw, bh = (int(v) for v in row[:4])
    score = float(row[14])

    problems = face_quality_problems(bw, bh, score, b.cfg)
    if problems:
        return None, problems

    m = int(FACE_MARGIN * max(bw, bh))
    x1, y1 = max(0, x - m), max(0, y - m)
    x2, y2 = min(w, x + bw + m), min(h, y + bh + m)
    if x2 <= x1 or y2 <= y1:
        return None, ["face crop fell outside the frame"]

    crop = frame_bgr[y1:y2, x1:x2]
    landmarks = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
    return FaceCrop(crop, frame_bgr, landmarks, score), []


@torch.inference_mode()
def embed_faces(b: Bundle, crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
    tensors = []
    for crop in crops_bgr:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (FACE_INPUT, FACE_INPUT), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensors.append(TF.normalize(t, list(IMAGENET_MEAN), list(IMAGENET_STD)))
    x = torch.stack(tensors).to(b.device)
    return b.encoder(x).cpu().numpy().astype(np.float32)


def _align_for_mobilefer(frame_bgr: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
    matrix, _ = cv2.estimateAffinePartial2D(landmarks, MOBILEFER_REFERENCE, method=cv2.LMEDS)
    if matrix is None:
        return None
    return cv2.warpAffine(
        frame_bgr,
        matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _softmax(scores: np.ndarray) -> np.ndarray:
    s = scores.astype(np.float64) - np.max(scores)
    e = np.exp(s)
    return e / np.sum(e)


def fer_probabilities(b: Bundle, crop_bgr: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    x = gray.astype(np.float32)[None, None]
    out = b.fer.run(None, {b.fer.get_inputs()[0].name: x})[0].reshape(-1)
    if out.shape[0] != len(FER_LABELS):
        raise RuntimeError(f"FERPlus returned {out.shape[0]} scores, expected {len(FER_LABELS)}")
    p = _softmax(out)
    return {label: float(p[i]) for i, label in enumerate(FER_LABELS)}


def mobilefer_label(b: Bundle, face: FaceCrop) -> str:
    aligned = _align_for_mobilefer(face.frame_bgr, face.landmarks)
    if aligned is None:
        raise RuntimeError("face alignment failed")
    img = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    blob = cv2.dnn.blobFromImage(img, 1.0, (112, 112), (0, 0, 0), swapRB=False, crop=False)
    b.mobilefer.setInput(blob, "data")
    out = b.mobilefer.forward("label")
    idx = int(np.asarray(out).reshape(-1)[0])
    if not 0 <= idx < len(MOBILEFER_LABELS):
        raise RuntimeError(f"MobileFaceNet label index {idx} out of range")
    return MOBILEFER_LABELS[idx]


def _expression_summary(b: Bundle, faces: List[FaceCrop], notes: List[str]) -> Optional[Dict]:
    """Advisory only. Never read by the decision logic."""
    if b.fer is None and b.mobilefer is None:
        return None

    fer_runs: List[Dict[str, float]] = []
    mobile_labels: List[str] = []
    for face in faces:
        if b.fer is not None:
            try:
                fer_runs.append(fer_probabilities(b, face.crop_bgr))
            except Exception as exc:
                notes.append(f"FERPlus failed on a frame: {exc}")
        if b.mobilefer is not None:
            try:
                mobile_labels.append(mobilefer_label(b, face))
            except Exception as exc:
                notes.append(f"MobileFaceNet failed on a frame: {exc}")

    result: Dict = {"fer_top": None, "fer_probs": {}, "mobile_top": None, "agree": None}
    if fer_runs:
        mean = {k: float(np.mean([run[k] for run in fer_runs])) for k in FER_LABELS}
        result["fer_probs"] = mean
        result["fer_top"] = max(mean, key=mean.get)
    if mobile_labels:
        result["mobile_top"] = majority_label(mobile_labels)
    if result["fer_top"] and result["mobile_top"]:
        result["agree"] = result["fer_top"] == result["mobile_top"]

    if result["fer_top"] is None and result["mobile_top"] is None:
        return None
    return result


def _face_stage(b: Bundle, frames: Sequence[np.ndarray], ev: Evidence) -> None:
    faces: List[FaceCrop] = []
    for frame in frames:
        ev.frames_checked += 1
        frame = limit_width(frame)
        face, problems = detect_face(b, frame)
        if face is None:
            ev.notes.extend(problems)
            continue
        faces.append(face)

    ev.faces_used = len(faces)
    if not faces:
        return

    # Same order as training: embed each frame, mean-pool, then classify.
    pooled = embed_faces(b, [f.crop_bgr for f in faces]).mean(axis=0)
    ev.face_prob = _positive_prob(b.face_model, b.face_scaler, b.face_index, pooled)
    ev.expression = _expression_summary(b, faces, ev.notes)


# =============================================================================
# Voice
# =============================================================================

def voice_features(audio: np.ndarray, sr: int = SR) -> Optional[np.ndarray]:
    """240-D vector: mean and std of MFCC, delta and delta-delta. Same as major.py."""
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if y.size == 0:
        return None
    peak = float(np.max(np.abs(y)))
    if peak > 1e-8:
        y = y / peak

    mfcc = librosa.feature.mfcc(
        y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    feats = np.concatenate(
        [
            mfcc.mean(axis=1), mfcc.std(axis=1),
            d1.mean(axis=1), d1.std(axis=1),
            d2.mean(axis=1), d2.std(axis=1),
        ]
    )
    return feats.astype(np.float32)


def _voice_stage(b: Bundle, audio: Optional[np.ndarray], sr: int, ev: Evidence) -> None:
    if audio is None or audio.size == 0:
        ev.notes.append("no audio available")
        return

    ev.audio_seconds = float(audio.size) / float(sr)
    if not audio_long_enough(ev.audio_seconds, b.cfg):
        ev.notes.append(f"under {b.cfg.min_audio_seconds:.0f}s of speech")
        return

    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
    if rms < SILENCE_RMS:
        ev.notes.append("no speech detected (audio too quiet)")
        return

    feats = voice_features(audio, sr)
    if feats is None:
        ev.notes.append("could not extract voice features")
        return
    ev.voice_prob = _positive_prob(b.voice_model, b.voice_scaler, b.voice_index, feats)


def decode_audio(path) -> Optional[np.ndarray]:
    """Mono float32 at SR, or None if the file has no audio stream."""
    with av.open(str(path)) as container:
        streams = container.streams.audio
        if not streams:
            return None
        resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks: List[np.ndarray] = []

        def take(frames) -> None:
            for f in frames or []:
                chunks.append(f.to_ndarray().reshape(-1))

        for packet in container.demux(streams[0]):
            for frame in packet.decode():
                take(resampler.resample(frame))
        take(resampler.resample(None))

    if not chunks:
        return None
    return np.concatenate(chunks).astype(np.float32)


# =============================================================================
# Scoring helpers
# =============================================================================

def _positive_prob(model, scaler, index: int, vector: np.ndarray) -> float:
    X = scaler.transform(np.asarray(vector, dtype=np.float32).reshape(1, -1))
    p = float(model.predict_proba(X)[0][index])
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise RuntimeError(f"invalid probability from model: {p!r}")
    return p


# =============================================================================
# Entry points
# =============================================================================

def analyze_video(b: Bundle, path) -> Evidence:
    """Upload mode. Same sampling as training: 5 evenly spaced frames."""
    ev = Evidence()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("The video could not be opened.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        ev.notes.append("frame count unavailable")

    frames: List[np.ndarray] = []
    for pos in sample_positions(total, b.cfg.n_face_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()

    _face_stage(b, frames, ev)
    _voice_stage(b, decode_audio(path), SR, ev)
    return ev


def analyze_live(
    b: Bundle,
    frames: Sequence[np.ndarray],
    audio: Optional[np.ndarray],
    sr: Optional[int],
) -> Evidence:
    """Live mode. Picks up to n_face_frames evenly spaced frames from the buffer."""
    ev = Evidence()
    positions = sample_positions(len(frames), b.cfg.n_face_frames)
    _face_stage(b, [frames[i] for i in positions], ev)
    if audio is None or sr is None:
        _voice_stage(b, None, SR, ev)
    else:
        _voice_stage(b, audio, sr, ev)
    return ev

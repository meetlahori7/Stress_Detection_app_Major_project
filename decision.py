"""Decision rules for the stress-proxy app.

Standard library only, so these rules can be unit-tested without torch,
OpenCV or Streamlit installed. Nothing here touches models or media.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

LABEL_STRESS = "stress"
LABEL_NO_STRESS = "no_stress"
LABEL_INCONCLUSIVE = "inconclusive"
LABEL_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Config:
    """Every tunable value in one place.

    Fusion weights and threshold are read from models/fusion_config.json so the
    trained configuration is the single source of truth.
    """

    face_weight: float = 0.40
    voice_weight: float = 0.60
    threshold: float = 0.50

    # Scores within +/- band of the threshold are reported as inconclusive.
    # The default is a guess. Tune it on validation data only.
    band: float = 0.05

    face_dim: int = 512
    voice_dim: int = 240

    # Must match training (major.py samples 5 frames per clip).
    n_face_frames: int = 5

    # Quality gates. Frames that fail them are skipped, not averaged in.
    min_face_px: int = 64
    min_face_score: float = 0.80

    min_audio_seconds: float = 2.0

    # Live mode: weight given to the newest value when smoothing.
    ema_alpha: float = 0.4

    def __post_init__(self) -> None:
        if abs(self.face_weight + self.voice_weight - 1.0) > 1e-6:
            raise ValueError("face_weight + voice_weight must equal 1.0")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not 0.0 <= self.band < 0.5:
            raise ValueError("band must be in [0, 0.5)")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.n_face_frames < 1:
            raise ValueError("n_face_frames must be at least 1")


_JSON_KEYS = {
    "face_weight": "face_weight",
    "voice_weight": "voice_weight",
    "threshold": "threshold",
    "face_dim": "face_feature_dimension",
    "voice_dim": "voice_feature_dimension",
}


def load_config(path) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fusion config not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    kwargs = {
        field_name: data[json_key]
        for field_name, json_key in _JSON_KEYS.items()
        if json_key in data
    }
    return Config(**kwargs)


@dataclass(frozen=True)
class Decision:
    label: str
    probability: Optional[float]
    reason: str


def fuse(p_face: Optional[float], p_voice: Optional[float], cfg: Config) -> Optional[float]:
    """Weighted average of the two branch probabilities.

    Returns None unless BOTH are present. The old app silently fell back to a
    single modality, which made the reported fusion score mean something else.
    """
    if p_face is None or p_voice is None:
        return None
    for name, value in (("face", p_face), ("voice", p_voice)):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid {name} probability: {value!r}")
    return cfg.face_weight * p_face + cfg.voice_weight * p_voice


def decide(p: Optional[float], cfg: Config) -> Decision:
    if p is None:
        return Decision(LABEL_UNAVAILABLE, None, "No decision was made.")
    if abs(p - cfg.threshold) < cfg.band:
        return Decision(
            LABEL_INCONCLUSIVE,
            p,
            f"Fusion score {p:.1%} is within ±{cfg.band:.0%} of the "
            f"{cfg.threshold:.0%} threshold, so the model does not make a call.",
        )
    if p >= cfg.threshold:
        return Decision(
            LABEL_STRESS,
            p,
            f"Fusion score {p:.1%} is at or above the {cfg.threshold:.0%} threshold.",
        )
    return Decision(
        LABEL_NO_STRESS,
        p,
        f"Fusion score {p:.1%} is below the {cfg.threshold:.0%} threshold.",
    )


def decide_live(p_face: Optional[float], p_voice: Optional[float], cfg: Config) -> Decision:
    """Live-mode decision with graceful fallback when one modality is missing."""
    if p_face is not None and p_voice is not None:
        return decide(fuse(p_face, p_voice, cfg), cfg)
    if p_face is None and p_voice is None:
        return Decision(
            LABEL_INCONCLUSIVE,
            None,
            "Collecting face and voice signals. Keep your face visible and continue speaking.",
        )

    if p_face is not None:
        base = decide(p_face, cfg)
        return Decision(
            base.label,
            base.probability,
            "Live fallback used face-only confidence because voice was unavailable. "
            + base.reason.replace("Fusion score", "Face score"),
        )

    base = decide(p_voice, cfg)
    return Decision(
        base.label,
        base.probability,
        "Live fallback used voice-only confidence because face was unavailable. "
        + base.reason.replace("Fusion score", "Voice score"),
    )


def ema(previous: Optional[float], value: Optional[float], alpha: float) -> Optional[float]:
    """Exponential moving average for live smoothing.

    A missing value resets the state to None, so a stale score is never shown
    as if it were current.
    """
    if value is None:
        return None
    if previous is None:
        return value
    return alpha * value + (1.0 - alpha) * previous


def sample_positions(total: int, k: int) -> list:
    """Evenly spaced indices, matching np.linspace(0, total-1, k, dtype=int).

    Duplicates are removed, so a short buffer yields fewer, distinct positions.
    """
    if total <= 0 or k <= 0:
        return []
    if k == 1 or total == 1:
        return [0]
    last = total - 1
    return sorted({int(i * last / (k - 1)) for i in range(k)})


def face_quality_problems(box_w: int, box_h: int, score: float, cfg: Config) -> list:
    problems = []
    if min(box_w, box_h) < cfg.min_face_px:
        problems.append(f"face smaller than {cfg.min_face_px}px")
    if score < cfg.min_face_score:
        problems.append(f"face detection confidence below {cfg.min_face_score:.2f}")
    return problems


def audio_long_enough(seconds: float, cfg: Config) -> bool:
    return seconds >= cfg.min_audio_seconds


def majority_label(labels: Iterable[Optional[str]]) -> Optional[str]:
    counts = Counter(label for label in labels if label)
    if not counts:
        return None
    return counts.most_common(1)[0][0]

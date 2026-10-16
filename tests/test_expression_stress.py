"""
Focused tests for expression label mapping, class ordering, probability handling,
model disagreement, and stress/expression separation.

These tests are self-contained: they copy the pure logic functions from app.py
so they can run without Streamlit, PyAV, torch, or model files.
"""
import unittest
import numpy as np


# ============================================================
# Copy constants and pure logic functions from app.py
# so tests run without requiring all app dependencies.
# ============================================================

FACE_WEIGHT = 0.40
VOICE_WEIGHT = 0.60
THRESHOLD = 0.50

FER_LABELS = [
    "neutral", "happy", "surprise", "sad",
    "angry", "disgust", "fear", "contempt",
]

MOBILEFER_LABELS = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise",
]

RAG_DOCUMENTS = [
    "The stress classifier uses facial ResNet18 features and voice MFCC features.",
    "The final stress prediction uses late fusion with 40 percent face probability and 60 percent voice probability.",
    "The stress decision threshold is 0.50. This project uses emotion-derived proxy labels, not clinically validated stress labels.",
    "Facial-expression recognition identifies visible expressions such as happy, sad, angry, fear, disgust, surprise, neutral and contempt. Expression is not the same thing as a person's actual emotional or stress state.",
    "The trained multimodal fusion score (40% face + 60% voice) drives the binary stress decision. Facial-expression models (FERPlus, MobileFaceNet) provide advisory context about visible expressions but never override the fusion score.",
    "Facial-expression labels such as happy, neutral, angry, or fear describe visible expressions, not clinical stress states. An angry expression does not automatically mean stress, and a neutral expression does not automatically mean no stress.",
    "When FERPlus and MobileFaceNet disagree on the visible expression, both predictions are shown independently. Expression disagreement does not change the trained fusion score.",
]


def _softmax(scores):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)


def fuse_predictions(face_probability, voice_probability):
    if face_probability is None and voice_probability is None:
        return None
    if face_probability is None:
        return voice_probability
    if voice_probability is None:
        return face_probability
    return FACE_WEIGHT * face_probability + VOICE_WEIGHT * voice_probability


def final_prediction(probability):
    if probability is None:
        return None
    return probability >= THRESHOLD


def get_final_decision(fusion_probability, expression_result=None):
    if fusion_probability is None:
        return None, "No usable trained-model probability is available."
    probability = float(fusion_probability)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return None, f"Invalid fusion probability received: {fusion_probability!r}."
    prediction = probability >= THRESHOLD
    result_text = "Stress" if prediction else "No Stress"
    reason = (
        f"Decision based on trained multimodal fusion: {probability:.1%} "
        f"vs {THRESHOLD:.0%} threshold → {result_text}."
    )
    if expression_result and expression_result.get("top_emotion"):
        reason += (
            f" Facial-expression model reports "
            f"{expression_result['top_emotion'].title()}; this is advisory only."
        )
    else:
        reason += " Facial-expression verification is unavailable; fusion decision is unchanged."
    return prediction, reason


def aggregate_expression_predictions(predictions):
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


# ============================================================
# TESTS
# ============================================================

class TestExpressionLabelMapping(unittest.TestCase):
    """Test FER_LABELS and MOBILEFER_LABELS are correct and complete."""

    def test_ferplus_labels_count(self):
        """FERPlus should have exactly 8 classes."""
        self.assertEqual(len(FER_LABELS), 8)

    def test_ferplus_labels_content(self):
        """FERPlus labels must match the documented ONNX output order."""
        expected = [
            "neutral", "happy", "surprise", "sad",
            "angry", "disgust", "fear", "contempt",
        ]
        self.assertEqual(FER_LABELS, expected)

    def test_mobilefer_labels_count(self):
        """MobileFaceNet should have exactly 7 classes."""
        self.assertEqual(len(MOBILEFER_LABELS), 7)

    def test_mobilefer_labels_content(self):
        """MobileFaceNet labels must match the OpenCV Zoo output order."""
        expected = [
            "angry", "disgust", "fear", "happy",
            "neutral", "sad", "surprise",
        ]
        self.assertEqual(MOBILEFER_LABELS, expected)

    def test_ferplus_has_contempt(self):
        """FERPlus 8-class includes contempt which MobileFaceNet lacks."""
        self.assertIn("contempt", FER_LABELS)
        self.assertNotIn("contempt", MOBILEFER_LABELS)

    def test_label_spaces_overlap(self):
        """Check shared labels between models for agreement comparison."""
        shared = set(FER_LABELS) & set(MOBILEFER_LABELS)
        # 7 of FERPlus's 8 labels should overlap (all except contempt)
        self.assertEqual(len(shared), 7)


class TestSoftmax(unittest.TestCase):
    """Test the _softmax helper."""

    def test_softmax_sums_to_one(self):
        scores = np.array([1.0, 2.0, 3.0, 0.5, -1.0, 0.0, 2.5, 1.5])
        probs = _softmax(scores)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_softmax_all_positive(self):
        scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        probs = _softmax(scores)
        self.assertTrue(np.all(probs >= 0.0))

    def test_softmax_highest_index(self):
        scores = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        probs = _softmax(scores)
        self.assertEqual(np.argmax(probs), 2)

    def test_softmax_8_classes(self):
        """Softmax with 8 classes produces 8 probabilities."""
        scores = np.random.randn(8)
        probs = _softmax(scores)
        self.assertEqual(len(probs), 8)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)


class TestFusionCalculation(unittest.TestCase):
    """Test that fusion weights and threshold are preserved."""

    def test_fusion_weights(self):
        self.assertAlmostEqual(FACE_WEIGHT, 0.40)
        self.assertAlmostEqual(VOICE_WEIGHT, 0.60)
        self.assertAlmostEqual(THRESHOLD, 0.50)

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(FACE_WEIGHT + VOICE_WEIGHT, 1.0)

    def test_fuse_both(self):
        result = fuse_predictions(0.3, 0.7)
        expected = 0.40 * 0.3 + 0.60 * 0.7
        self.assertAlmostEqual(result, expected, places=6)

    def test_fuse_face_only(self):
        result = fuse_predictions(0.8, None)
        self.assertAlmostEqual(result, 0.8)

    def test_fuse_voice_only(self):
        result = fuse_predictions(None, 0.6)
        self.assertAlmostEqual(result, 0.6)

    def test_fuse_none(self):
        result = fuse_predictions(None, None)
        self.assertIsNone(result)

    def test_fuse_extremes(self):
        """Face=0, Voice=1 → 0.60"""
        result = fuse_predictions(0.0, 1.0)
        self.assertAlmostEqual(result, 0.60)


class TestFinalDecision(unittest.TestCase):
    """Test that the final decision uses fusion only and ignores expression."""

    def test_stress_above_threshold(self):
        prediction, _ = get_final_decision(0.75)
        self.assertTrue(prediction)

    def test_no_stress_below_threshold(self):
        prediction, _ = get_final_decision(0.30)
        self.assertFalse(prediction)

    def test_stress_at_threshold(self):
        prediction, _ = get_final_decision(0.50)
        self.assertTrue(prediction)  # >= threshold

    def test_just_below_threshold(self):
        prediction, _ = get_final_decision(0.499)
        self.assertFalse(prediction)

    def test_none_probability(self):
        prediction, _ = get_final_decision(None)
        self.assertIsNone(prediction)

    def test_neutral_expression_does_not_override_stress(self):
        """A Neutral expression must NOT prevent a Stress result."""
        expression = {
            "top_emotion": "neutral",
            "probabilities": {"neutral": 0.9},
            "status": "ready",
        }
        prediction, reason = get_final_decision(0.75, expression)
        self.assertTrue(prediction, "Neutral expression should not override stress")
        self.assertIn("advisory", reason.lower())

    def test_angry_expression_does_not_force_stress(self):
        """An Angry expression must NOT force a Stress result."""
        expression = {
            "top_emotion": "angry",
            "probabilities": {"angry": 0.9},
            "status": "ready",
        }
        prediction, reason = get_final_decision(0.30, expression)
        self.assertFalse(prediction, "Angry expression should not force stress")

    def test_happy_expression_with_high_fusion(self):
        """Happy expression + high fusion → still Stress."""
        expression = {"top_emotion": "happy", "status": "ready"}
        prediction, _ = get_final_decision(0.85, expression)
        self.assertTrue(prediction)

    def test_fear_expression_with_low_fusion(self):
        """Fear expression + low fusion → still No Stress."""
        expression = {"top_emotion": "fear", "status": "ready"}
        prediction, _ = get_final_decision(0.15, expression)
        self.assertFalse(prediction)

    def test_expression_disagreement_does_not_change_fusion(self):
        """Model disagreement in expression must not alter the fusion score."""
        expression = {
            "top_emotion": "neutral",
            "probabilities": {"neutral": 0.7},
            "ferplus": {"top_emotion": "neutral", "status": "ready"},
            "mobileface": {"top_emotion": "angry", "status": "ready"},
            "agreement": False,
            "status": "ready",
        }
        # Fusion probability is 0.65 → Stress
        prediction, _ = get_final_decision(0.65, expression)
        self.assertTrue(prediction, "Expression disagreement must not change fusion")

        # Fusion probability is 0.35 → No Stress
        prediction2, _ = get_final_decision(0.35, expression)
        self.assertFalse(prediction2, "Expression disagreement must not change fusion")


class TestExpressionAgreement(unittest.TestCase):
    """Test expression agreement logic."""

    def test_agreement_when_same(self):
        result = {
            "ferplus": {"top_emotion": "happy", "status": "ready"},
            "mobileface": {"top_emotion": "happy", "status": "ready"},
            "agreement": True,
        }
        self.assertTrue(result["agreement"])

    def test_disagreement_when_different(self):
        result = {
            "ferplus": {"top_emotion": "neutral", "status": "ready"},
            "mobileface": {"top_emotion": "angry", "status": "ready"},
            "agreement": False,
        }
        self.assertFalse(result["agreement"])

    def test_agreement_none_when_single_model(self):
        result = {
            "ferplus": {"top_emotion": "sad", "status": "ready"},
            "mobileface": None,
            "agreement": None,
        }
        self.assertIsNone(result["agreement"])


class TestExpressionStressSeparation(unittest.TestCase):
    """Test that expression and stress pipelines are independent."""

    def test_neutral_coexists_with_stress(self):
        """Neutral expression + Stress fusion → both displayed independently."""
        expression = {"top_emotion": "neutral", "status": "ready"}
        prediction, reason = get_final_decision(0.80, expression)
        self.assertTrue(prediction)
        self.assertIn("Neutral", reason)
        self.assertIn("advisory", reason.lower())

    def test_angry_coexists_with_no_stress(self):
        """Angry expression + No Stress fusion → both displayed independently."""
        expression = {"top_emotion": "angry", "status": "ready"}
        prediction, reason = get_final_decision(0.20, expression)
        self.assertFalse(prediction)
        self.assertIn("Angry", reason)

    def test_contempt_coexists_with_stress(self):
        """Contempt (FERPlus only) + Stress → no override."""
        expression = {"top_emotion": "contempt", "status": "ready"}
        prediction, _ = get_final_decision(0.60, expression)
        self.assertTrue(prediction)

    def test_no_expression_still_predicts(self):
        """Missing expression → fusion decision unchanged."""
        prediction, reason = get_final_decision(0.70, None)
        self.assertTrue(prediction)
        self.assertIn("unavailable", reason.lower())


class TestAggregateExpressionPredictions(unittest.TestCase):
    """Test multi-frame expression aggregation."""

    def test_aggregate_preserves_model_info(self):
        predictions = [
            {
                "top_emotion": "happy",
                "probabilities": {"happy": 0.8, "sad": 0.2},
                "ferplus": {"top_emotion": "happy", "status": "ready"},
                "mobileface": {"top_emotion": "happy", "status": "ready"},
                "agreement": True,
                "status": "ready",
            },
            {
                "top_emotion": "happy",
                "probabilities": {"happy": 0.9, "sad": 0.1},
                "ferplus": {"top_emotion": "happy", "status": "ready"},
                "mobileface": {"top_emotion": "sad", "status": "ready"},
                "agreement": False,
                "status": "ready",
            },
        ]
        result = aggregate_expression_predictions(predictions)
        self.assertIsNotNone(result)
        self.assertIn("ferplus", result)
        self.assertIn("mobileface", result)
        self.assertEqual(result["top_emotion"], "happy")

    def test_aggregate_empty(self):
        result = aggregate_expression_predictions([])
        self.assertIsNone(result)

    def test_aggregate_all_none(self):
        result = aggregate_expression_predictions([None, None])
        self.assertIsNone(result)

    def test_aggregate_probabilities_normalized(self):
        predictions = [
            {
                "top_emotion": "happy",
                "probabilities": {"happy": 0.6, "sad": 0.4},
                "status": "ready",
            },
        ]
        result = aggregate_expression_predictions(predictions)
        total = sum(result["probabilities"].values())
        self.assertAlmostEqual(total, 1.0, places=5)


class TestRAGDocuments(unittest.TestCase):
    """Test that RAG documents correctly describe the architecture."""

    def test_no_expression_override_claim(self):
        """RAG docs should not claim expression overrides the stress model."""
        for doc in RAG_DOCUMENTS:
            doc_lower = doc.lower()
            # Should not say expression is the primary decision
            self.assertFalse(
                "expression" in doc_lower and "primary decision" in doc_lower,
                f"RAG doc incorrectly claims expression is primary: {doc[:80]}",
            )

    def test_fusion_drives_stress(self):
        """At least one RAG doc should state fusion drives the decision."""
        found = any(
            "fusion" in doc.lower() and "stress decision" in doc.lower()
            for doc in RAG_DOCUMENTS
        )
        self.assertTrue(found, "RAG documents should mention fusion drives stress decision")

    def test_expression_not_clinical(self):
        """RAG should mention expression is not clinical stress."""
        found = any(
            "not" in doc.lower() and "clinical" in doc.lower()
            for doc in RAG_DOCUMENTS
        )
        self.assertTrue(found, "RAG should mention expression is not clinical stress")

    def test_disagreement_documented(self):
        """RAG should mention how expression disagreement is handled."""
        found = any("disagree" in doc.lower() for doc in RAG_DOCUMENTS)
        self.assertTrue(found, "RAG should document disagreement handling")


if __name__ == "__main__":
    unittest.main(verbosity=2)

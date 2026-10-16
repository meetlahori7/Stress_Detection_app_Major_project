"""Unit tests for the decision rules. Run from the project folder:

    python -m unittest discover tests -v
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from decision import (  # noqa: E402
    LABEL_INCONCLUSIVE,
    LABEL_NO_STRESS,
    LABEL_STRESS,
    LABEL_UNAVAILABLE,
    Config,
    audio_long_enough,
    decide,
    ema,
    face_quality_problems,
    fuse,
    load_config,
    majority_label,
    sample_positions,
)


class TestFusion(unittest.TestCase):
    def test_uses_config_weights(self):
        cfg = Config()
        self.assertAlmostEqual(fuse(0.3, 0.7, cfg), 0.4 * 0.3 + 0.6 * 0.7)

    def test_requires_both_modalities(self):
        cfg = Config()
        self.assertIsNone(fuse(0.5, None, cfg))
        self.assertIsNone(fuse(None, 0.5, cfg))

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            fuse(1.5, 0.2, Config())

    def test_rejects_nan(self):
        with self.assertRaises(ValueError):
            fuse(float("nan"), 0.2, Config())


class TestDecision(unittest.TestCase):
    def test_threshold_boundary_is_stress_when_no_band(self):
        cfg = Config(band=0.0)
        self.assertEqual(decide(0.5, cfg).label, LABEL_STRESS)

    def test_just_below_threshold_is_no_stress_when_no_band(self):
        cfg = Config(band=0.0)
        self.assertEqual(decide(0.499, cfg).label, LABEL_NO_STRESS)

    def test_scores_near_threshold_are_inconclusive(self):
        cfg = Config(band=0.05)
        self.assertEqual(decide(0.52, cfg).label, LABEL_INCONCLUSIVE)
        self.assertEqual(decide(0.47, cfg).label, LABEL_INCONCLUSIVE)

    def test_clear_scores_are_decided(self):
        cfg = Config(band=0.05)
        self.assertEqual(decide(0.80, cfg).label, LABEL_STRESS)
        self.assertEqual(decide(0.20, cfg).label, LABEL_NO_STRESS)

    def test_missing_score_is_unavailable(self):
        self.assertEqual(decide(None, Config()).label, LABEL_UNAVAILABLE)


class TestConfig(unittest.TestCase):
    def test_weights_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            Config(face_weight=0.5, voice_weight=0.6)

    def test_load_config_reads_trained_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fusion_config.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    '{"face_weight": 0.4, "voice_weight": 0.6, "threshold": 0.5,'
                    ' "face_feature_dimension": 512, "voice_feature_dimension": 240}'
                )
            cfg = load_config(path)
        self.assertEqual(cfg.face_dim, 512)
        self.assertEqual(cfg.voice_dim, 240)
        self.assertEqual(cfg.threshold, 0.5)

    def test_load_config_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_config(os.path.join(tempfile.gettempdir(), "does_not_exist_cfg.json"))


class TestSmoothing(unittest.TestCase):
    def test_first_value_passes_through(self):
        self.assertEqual(ema(None, 0.6, 0.4), 0.6)

    def test_blends_with_previous(self):
        self.assertAlmostEqual(ema(0.6, 1.0, 0.5), 0.8)

    def test_missing_value_resets_state(self):
        # A stale score must never be displayed as current.
        self.assertIsNone(ema(0.8, None, 0.4))


class TestSamplingAndQuality(unittest.TestCase):
    def test_sample_positions_match_linspace(self):
        self.assertEqual(sample_positions(100, 5), [0, 24, 49, 74, 99])

    def test_short_buffer_has_no_duplicates(self):
        self.assertEqual(sample_positions(3, 5), [0, 1, 2])

    def test_empty_buffer(self):
        self.assertEqual(sample_positions(0, 5), [])

    def test_small_face_is_rejected(self):
        problems = face_quality_problems(40, 200, 0.95, Config(min_face_px=64))
        self.assertTrue(any("smaller" in p for p in problems))

    def test_low_confidence_face_is_rejected(self):
        problems = face_quality_problems(200, 200, 0.5, Config(min_face_score=0.8))
        self.assertTrue(any("confidence" in p for p in problems))

    def test_good_face_passes(self):
        self.assertEqual(face_quality_problems(200, 200, 0.95, Config()), [])

    def test_audio_length_gate(self):
        cfg = Config(min_audio_seconds=2.0)
        self.assertTrue(audio_long_enough(2.0, cfg))
        self.assertFalse(audio_long_enough(1.9, cfg))

    def test_majority_label(self):
        self.assertEqual(majority_label(["happy", "sad", "happy"]), "happy")
        self.assertIsNone(majority_label([]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

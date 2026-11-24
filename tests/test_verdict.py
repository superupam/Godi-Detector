"""Tests for utils.decide_verdict — the shared verdict synthesis.

Engine 2 result dicts are built by hand so the tests describe the rule table
without touching the network.
"""

import pytest

from utils import decide_verdict


def fact(status, confidence=0.9, source="reuters.com"):
    """Build the subset of fact_checker's result dict decide_verdict reads."""
    return {
        "status": status,
        "confidence": confidence,
        "top_source": source,
        "top_snippet": f"Evidence snippet for {status}.",
    }


# --- Engine 2 overrides Engine 1 -------------------------------------------


def test_contradicted_is_debunked_even_in_formal_prose():
    decision = decide_verdict(99.0, 1.0, fact("CONTRADICTED"))
    assert decision["verdict"] == "DEBUNKED / FACTUALLY FALSE"
    assert decision["category"] == "fake"
    assert decision["label"] == "Factually Debunked"


def test_contradicted_confidence_has_a_96_floor():
    assert decide_verdict(90.0, 10.0, fact("CONTRADICTED", 0.41))["confidence"] == 96.0


def test_contradicted_confidence_above_the_floor_is_kept():
    assert decide_verdict(50.0, 50.0, fact("CONTRADICTED", 0.989))["confidence"] == 98.9


def test_corroborated_confidence_has_an_88_floor():
    decision = decide_verdict(80.0, 20.0, fact("CORROBORATED", 0.62))
    assert decision["verdict"] == "VERIFIED REAL NEWS"
    assert decision["category"] == "real"
    assert decision["confidence"] == 88.0
    assert "reuters.com" in decision["explanation"]


def test_corroborated_confidence_above_the_floor_is_kept():
    assert decide_verdict(40.0, 60.0, fact("CORROBORATED", 0.94))["confidence"] == 94.0


# --- Partial evidence: the style split at 60 -------------------------------


@pytest.mark.parametrize("real_prob", [60.0, 75.0, 100.0])
def test_partial_evidence_with_formal_style_is_likely_real(real_prob):
    decision = decide_verdict(real_prob, 100.0 - real_prob, fact("PARTIALLY_CORROBORATED"))
    assert decision["verdict"] == "LIKELY REAL (PARTIALLY CORROBORATED)"
    assert decision["category"] == "real"
    assert decision["confidence"] == 72.0
    assert decision["label"] == "Partial Evidence"


@pytest.mark.parametrize("real_prob", [0.0, 45.0, 59.9])
def test_partial_evidence_with_formal_style_just_under_60_is_caution(real_prob):
    decision = decide_verdict(real_prob, 100.0 - real_prob, fact("PARTIALLY_CORROBORATED"))
    assert decision["verdict"] == "UNVERIFIED / REQUIRES CAUTION"
    assert decision["category"] == "warn"
    assert decision["confidence"] == 58.0


# --- No usable evidence ----------------------------------------------------


def test_no_fact_result_and_clickbait_style_is_fake():
    decision = decide_verdict(20.0, 70.0, None)
    assert decision["verdict"] == "LIKELY FAKE NEWS / CLICKBAIT"
    assert decision["category"] == "fake"
    assert decision["confidence"] == 70.0


def test_clickbait_check_wins_over_the_formal_tone_branch():
    decision = decide_verdict(90.0, 80.0, None)
    assert decision["category"] == "fake"
    assert decision["confidence"] == 80.0


def test_formal_tone_without_evidence_is_an_unverified_claim():
    decision = decide_verdict(75.0, 25.0, None)
    assert decision["verdict"] == "UNVERIFIED CLAIM (FORMAL JOURNALISTIC TONE)"
    assert decision["category"] == "warn"
    assert decision["confidence"] == 65.0


def test_neutral_style_without_evidence_is_ambiguous():
    decision = decide_verdict(50.0, 50.0, None)
    assert decision["verdict"] == "AMBIGUOUS / UNVERIFIED"
    assert decision["category"] == "warn"
    assert decision["confidence"] == 50.0


@pytest.mark.parametrize("status", ["NO_EVIDENCE", "ERROR", "SOMETHING_NEW"])
def test_unknown_engine2_status_falls_through_to_style(status):
    decision = decide_verdict(10.0, 90.0, fact(status))
    assert decision["category"] == "fake"
    assert decision["confidence"] == 90.0


# --- Streamlit strictness slider -------------------------------------------


def test_corroborated_below_the_slider_is_flagged_not_verified():
    decision = decide_verdict(60.0, 40.0, fact("CORROBORATED", 0.80), min_confidence=85)
    assert decision["verdict"] == "PARTIALLY VERIFIED (BELOW STRICTNESS THRESHOLD)"
    assert decision["category"] == "warn"
    assert decision["confidence"] == 80.0
    assert "85%" in decision["explanation"]


def test_corroborated_exactly_at_the_slider_still_verifies():
    decision = decide_verdict(60.0, 40.0, fact("CORROBORATED", 0.85), min_confidence=85)
    assert decision["verdict"] == "VERIFIED REAL NEWS"


def test_slider_overrides_the_clickbait_threshold():
    decision = decide_verdict(10.0, 75.0, None, min_confidence=85)
    assert decision["verdict"] == "AMBIGUOUS / UNVERIFIED"
    assert decision["confidence"] == 50.0


def test_slider_overrides_the_formal_tone_threshold():
    decision = decide_verdict(72.0, 28.0, None, min_confidence=85)
    assert decision["verdict"] == "AMBIGUOUS / UNVERIFIED"


# --- Shape ---------------------------------------------------------------


def test_result_has_exactly_the_documented_keys():
    decision = decide_verdict(50.0, 50.0, None)
    assert set(decision) == {"verdict", "category", "confidence", "explanation", "label"}


def test_confidence_is_rounded_to_one_decimal():
    assert decide_verdict(20.0, 84.267, None)["confidence"] == 84.3

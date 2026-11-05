"""Drift monitoring contracts: population stability index on match scores."""

from __future__ import annotations

import pytest

from fuzzymatch.monitoring import drift_report


class TestDriftReport:
    def test_identical_distribution_is_stable(self) -> None:
        reference = [0.1] * 150 + [0.5] * 100 + [0.9] * 60
        report = drift_report(reference, list(reference))
        assert report["psi"] < 0.01
        assert report["warning"] is False

    def test_shifted_distribution_warns(self) -> None:
        reference = [0.05] * 200 + [0.15] * 120
        current = [0.75] * 200 + [0.85] * 120
        report = drift_report(reference, current)
        assert report["psi"] > 0.25
        assert report["warning"] is True

    def test_moderate_shift_below_warning_threshold(self) -> None:
        reference = [i / 400 for i in range(400)]
        current = [min(1.0, i / 400 + 0.02) for i in range(400)]
        report = drift_report(reference, current)
        assert report["psi"] < 0.25

    def test_keys_present(self) -> None:
        report = drift_report([0.1, 0.9] * 200, [0.2, 0.8] * 200)
        assert set(report) == {
            "status",
            "psi",
            "js_divergence",
            "warning",
            "n_reference",
            "n_current",
        }

    def test_small_windows_report_insufficient_status(self) -> None:
        report = drift_report([0.5, 0.6], [0.4, 0.7])
        assert report["status"] == "insufficient_data"
        assert report["psi"] is None
        assert report["js_divergence"] is None
        assert report["warning"] is False
        assert report["n_reference"] == 2

    def test_js_divergence_present_and_bounded(self) -> None:
        ref = [i / 400 for i in range(400)]
        cur = [(i / 400) ** 0.5 for i in range(400)]
        identical = drift_report(ref, list(ref))
        shifted = drift_report(ref, cur)
        assert identical["status"] == "ok"
        assert 0.0 <= identical["js_divergence"] <= 1.0
        assert shifted["js_divergence"] > identical["js_divergence"]
        assert shifted["psi"] is not None and shifted["psi"] >= 0.0

    def test_empty_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            drift_report([], [0.5])
        with pytest.raises(ValueError, match="empty"):
            drift_report([0.5], [])

    def test_bins_below_two_raises(self) -> None:
        with pytest.raises(ValueError, match="bins"):
            drift_report([0.1, 0.9], [0.2, 0.8], bins=1)

    def test_deterministic(self) -> None:
        a = drift_report([0.1, 0.4, 0.7], [0.2, 0.5])
        b = drift_report([0.1, 0.4, 0.7], [0.2, 0.5])
        assert a == b

    def test_custom_warning_threshold_respected(self) -> None:
        # Moderate shift: trips the 0.25 default, not a relaxed 10.0 bar.
        reference = [i / 400 for i in range(400)]
        current = [min(1.0, i / 400 + 0.35) for i in range(400)]
        default_report = drift_report(reference, current)
        relaxed = drift_report(reference, current, psi_warning=10.0)
        assert default_report["warning"] is True
        assert relaxed["warning"] is False

    def test_discrete_quantile_tied_simplex_conservation(self) -> None:
        """Verify probability simplex conservation when quantile edges collapse."""
        reference = [0.0] * 250 + [1.0] * 150
        current = [0.0] * 200 + [1.0] * 200
        report = drift_report(reference, current, bins=10)
        assert report["status"] == "ok"
        assert report["psi"] is not None and report["psi"] >= 0.0
        assert report["js_divergence"] is not None and report["js_divergence"] >= 0.0

        # Exact self-drift must have PSI and JSD identically zero
        self_report = drift_report(reference, list(reference), bins=10)
        assert self_report["psi"] == 0.0
        assert self_report["js_divergence"] == 0.0

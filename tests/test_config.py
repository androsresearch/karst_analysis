"""Tests for the YAML config loader.

Covers: loading the canonical default; deep-merging user overrides;
type validation; missing-file errors; ledger-params extraction;
unknown-key warnings.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from karst_analysis.config import (
    ConfigError, default_config_path, load_config, params_for_run_ledger,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = ROOT / "config" / "pipeline_default.yml"


pytestmark = pytest.mark.skipif(
    not DEFAULT_CFG.exists(),
    reason="config/pipeline_default.yml is required for these tests",
)


class TestLoadDefaults:
    def test_default_path_resolves(self):
        p = default_config_path()
        assert p.name == "pipeline_default.yml"
        assert p.is_file()

    def test_load_default_only(self):
        cfg = load_config()  # path=None → defaults only
        assert cfg["campaign"] == "2022_02"
        assert cfg["preprocessing"]["savgol"]["window"] == 11
        assert cfg["preprocessing"]["lowess"]["frac"] == 0.05
        assert cfg["breakpoints"]["max_breakpoints"] == 10
        assert cfg["robustness"]["delta_m"] == 0.5

    def test_default_has_all_required_sections(self):
        cfg = load_config()
        for k in ("campaign", "preprocessing", "breakpoints", "robustness"):
            assert k in cfg


class TestDeepMerge:
    def test_partial_override_savgol_window(self, tmp_path):
        user = tmp_path / "user.yml"
        user.write_text(
            "preprocessing:\n"
            "  savgol:\n"
            "    window: 21\n"
        )
        cfg = load_config(user)
        # Override applied
        assert cfg["preprocessing"]["savgol"]["window"] == 21
        # Sibling keys preserved from default
        assert cfg["preprocessing"]["savgol"]["order"] == 3
        # Unrelated sections preserved
        assert cfg["preprocessing"]["lowess"]["frac"] == 0.05
        assert cfg["robustness"]["delta_m"] == 0.5

    def test_override_top_level_campaign(self, tmp_path):
        user = tmp_path / "user.yml"
        # Quote the value because YAML parses 2023_07 as int 202307
        # (underscore is a digit separator). For real configs the user
        # MUST quote campaign tags that are date-like.
        user.write_text("campaign: \"2023_07\"\n")
        cfg = load_config(user)
        assert cfg["campaign"] == "2023_07"
        # Other sections still come from defaults
        assert cfg["preprocessing"]["savgol"]["window"] == 11

    def test_override_list_replaces_not_merges(self, tmp_path):
        """Lists should be replaced, not concatenated."""
        user = tmp_path / "user.yml"
        user.write_text(
            "robustness:\n"
            "  sensitivity_deltas_m: [0.4, 0.6]\n"
        )
        cfg = load_config(user)
        assert cfg["robustness"]["sensitivity_deltas_m"] == [0.4, 0.6]
        # Other robustness keys still default
        assert cfg["robustness"]["delta_m"] == 0.5

    def test_empty_user_config(self, tmp_path):
        """An empty file should load identical to defaults-only."""
        user = tmp_path / "empty.yml"
        user.write_text("")
        cfg_user = load_config(user)
        cfg_default = load_config()
        assert cfg_user == cfg_default


class TestValidation:
    def test_wrong_type_raises(self, tmp_path):
        user = tmp_path / "bad.yml"
        user.write_text(
            "preprocessing:\n"
            "  savgol:\n"
            "    window: \"eleven\"\n"
        )
        with pytest.raises(ConfigError, match="must be int"):
            load_config(user)

    def test_bool_where_int_expected_raises(self, tmp_path):
        user = tmp_path / "bad.yml"
        user.write_text(
            "preprocessing:\n"
            "  savgol:\n"
            "    window: true\n"
        )
        with pytest.raises(ConfigError, match="must be int.*bool"):
            load_config(user)

    def test_int_where_bool_expected_raises(self, tmp_path):
        user = tmp_path / "bad.yml"
        user.write_text(
            "preprocessing:\n"
            "  savgol:\n"
            "    segmented: 1\n"
        )
        with pytest.raises(ConfigError, match="must be bool"):
            load_config(user)

    def test_unknown_key_is_warning_not_error(self, tmp_path):
        """Typos at top-level should warn, not abort."""
        user = tmp_path / "typo.yml"
        user.write_text(
            "preprocessing:\n"
            "  savgol:\n"
            "    windwo: 21\n"  # typo on purpose
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = load_config(user)
        # Default value still in place because typo'd key was ignored
        assert cfg["preprocessing"]["savgol"]["window"] == 11
        # Warning was emitted
        unknowns = [w for w in caught if "Unknown key" in str(w.message)]
        assert len(unknowns) >= 1
        assert "windwo" in str(unknowns[0].message)

    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.yml"
        with pytest.raises(ConfigError, match="not found"):
            load_config(missing)

    def test_non_dict_top_level_raises(self, tmp_path):
        bad = tmp_path / "scalar.yml"
        bad.write_text("just a string\n")
        with pytest.raises(ConfigError, match="must contain a YAML mapping"):
            load_config(bad)


class TestParamsForRunLedger:
    def test_savgol_includes_segmented_extras(self):
        cfg = load_config()
        p = params_for_run_ledger(cfg, "preprocessing", "savgol")
        assert p["method"] == "savgol"
        assert p["window"] == 11
        assert p["order"] == 3
        assert p["segmented"] is True
        assert "gradient_factor" in p
        assert "min_gradient_threshold" in p
        # Should NOT contain lowess params
        assert "frac" not in p
        assert "iter" not in p

    def test_lowess_includes_pava(self):
        cfg = load_config()
        p = params_for_run_ledger(cfg, "preprocessing", "lowess")
        assert p["method"] == "lowess"
        assert p["frac"] == 0.05
        assert p["iter"] == 2
        assert p["pava"] is True
        assert p["degree"] == 1
        # Should NOT contain savgol params
        assert "window" not in p
        assert "order" not in p

    def test_breakpoints_subset(self):
        cfg = load_config()
        p = params_for_run_ledger(cfg, "breakpoints")
        assert p["max_breakpoints"] == 10
        assert p["use_log10"] is True

    def test_robustness_subset(self):
        cfg = load_config()
        p = params_for_run_ledger(cfg, "robustness")
        assert p["delta_m"] == 0.5
        assert p["sensitivity_deltas_m"] == [0.3, 0.5, 1.0]

    def test_unknown_method_raises(self):
        cfg = load_config()
        with pytest.raises(ValueError, match="Unknown method"):
            params_for_run_ledger(cfg, "preprocessing", method="banana")

    def test_unknown_stage_raises(self):
        cfg = load_config()
        with pytest.raises(ValueError, match="Unknown stage"):
            params_for_run_ledger(cfg, "magic")

    def test_preprocessing_requires_method(self):
        cfg = load_config()
        with pytest.raises(ValueError, match="method must be specified"):
            params_for_run_ledger(cfg, "preprocessing")


class TestConvergenceSecCaliperValidation:
    """Domain-specific validation for the convergence.sec_caliper block."""

    def test_default_loads(self):
        cfg = load_config()
        sub = cfg["convergence"]["sec_caliper"]
        assert sub["matching_rule"] == "hybrid"
        assert sub["tolerance_m"] == 0.5
        assert sub["sec_agreement_min"] == 3
        assert sub["severity_weights"] == {"mild": 1, "moderate": 2, "severe": 3}
        assert sub["best_match_priority"] == "overlap_then_distance"
        assert sub["unmatched_zones_min_severity"] == "moderate"
        assert sub["caliper_severity_filter"] is None
        assert sub["run_tag"] == "convergence_sec_caliper_v1"

    def test_invalid_matching_rule_raises(self, tmp_path):
        user = tmp_path / "u.yml"
        user.write_text(
            "convergence:\n"
            "  sec_caliper:\n"
            "    matching_rule: bogus\n"
        )
        with pytest.raises(ConfigError, match="matching_rule must be one of"):
            load_config(user)

    def test_invalid_priority_raises(self, tmp_path):
        user = tmp_path / "u.yml"
        user.write_text(
            "convergence:\n"
            "  sec_caliper:\n"
            "    best_match_priority: nope\n"
        )
        with pytest.raises(ConfigError, match="best_match_priority must be one of"):
            load_config(user)

    def test_severity_weights_missing_key_in_partial_default_raises(
        self, tmp_path,
    ):
        # If we override the whole defaults to a partial dict (which
        # only happens if someone supplies a custom default_path), the
        # validator should still catch missing severity keys.
        bad_default = tmp_path / "bad_default.yml"
        # Minimal full config with severity_weights missing 'severe'.
        bad_default.write_text("""
campaign: "2022_02"
preprocessing:
  methods: ["savgol", "lowess"]
  apply_depth_adjustment: false
  depth_adjustment: 0.272
  depth_adjustment_method: "TOM"
  apply_monotonic_descent_filter: true
  monotonic_descent_tolerance: 0.002
  dz: null
  dz_method: "percentile95"
  apply_log10: true
  savgol: {window: 11, order: 3, segmented: true, gradient_factor: 20.0, min_gradient_threshold: 1000.0}
  lowess: {frac: 0.05, degree: 1, n_robust_iter: 2, apply_pava: true}
breakpoints: {max_breakpoints: 10, n_trials: 3, tolerance: 1.0e-5, min_distance: 0.01, use_log10: true}
robustness: {delta_m: 0.5, sensitivity_deltas_m: [0.3, 0.5, 1.0], n_min: 1, n_max: 10, smoothings: ["savgol", "lowess"]}
convergence:
  sec_caliper:
    sec_agreement_min: 3
    caliper_severity_filter: null
    matching_rule: "hybrid"
    tolerance_m: 0.5
    best_match_priority: "overlap_then_distance"
    severity_weights: {mild: 1, moderate: 2}   # severe missing!
    unmatched_zones_min_severity: "moderate"
    run_tag: "test_v1"
ert:
  breakpoints: {max_breakpoints: 15, tolerance: 1.0e-5, min_distance: 0.01, start_seed: 0, max_seed_attempts: 20}
  bot_mz_rho_threshold: 25.0
""")
        with pytest.raises(ConfigError, match="missing keys"):
            load_config(default_path=bad_default)

    def test_severity_weights_unknown_key_raises(self, tmp_path):
        user = tmp_path / "u.yml"
        user.write_text(
            "convergence:\n"
            "  sec_caliper:\n"
            "    severity_weights:\n"
            "      mild: 1\n"
            "      moderate: 2\n"
            "      severe: 3\n"
            "      catastrophic: 99\n"
        )
        with pytest.raises(ConfigError, match="unknown keys"):
            load_config(user)

    def test_negative_tolerance_raises(self, tmp_path):
        user = tmp_path / "u.yml"
        user.write_text(
            "convergence:\n"
            "  sec_caliper:\n"
            "    tolerance_m: -1.0\n"
        )
        with pytest.raises(ConfigError, match="tolerance_m must be >= 0"):
            load_config(user)

    def test_user_can_override_tolerance(self, tmp_path):
        user = tmp_path / "u.yml"
        user.write_text(
            "convergence:\n"
            "  sec_caliper:\n"
            "    tolerance_m: 1.0\n"
        )
        cfg = load_config(user)
        assert cfg["convergence"]["sec_caliper"]["tolerance_m"] == 1.0
        # Other keys still default
        assert cfg["convergence"]["sec_caliper"]["matching_rule"] == "hybrid"

    def test_caliper_severity_filter_with_unknown_severity_raises(self, tmp_path):
        user = tmp_path / "u.yml"
        user.write_text(
            "convergence:\n"
            "  sec_caliper:\n"
            "    caliper_severity_filter: ['mild', 'apocalyptic']\n"
        )
        with pytest.raises(ConfigError, match="unknown severities"):
            load_config(user)

    def test_unmatched_min_severity_null_accepted(self, tmp_path):
        user = tmp_path / "u.yml"
        user.write_text(
            "convergence:\n"
            "  sec_caliper:\n"
            "    unmatched_zones_min_severity: null\n"
        )
        cfg = load_config(user)
        assert cfg["convergence"]["sec_caliper"]["unmatched_zones_min_severity"] is None

    def test_params_for_run_ledger_convergence_sec_caliper(self):
        cfg = load_config()
        p = params_for_run_ledger(cfg, "convergence_sec_caliper")
        assert p["method"] == "convergence_sec_caliper"
        assert p["matching_rule"] == "hybrid"
        assert p["tolerance_m"] == 0.5
        assert p["sec_agreement_min"] == 3

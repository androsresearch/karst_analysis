"""Convention-aware figure-output path resolver  (v13).

Centralises the rule "where do figures of technique <X> for campaign(s)
<Y> go?" so each module/script does not re-implement it.

Convention (v13)
----------------
The project distinguishes three classes of techniques by what kind of
campaign concept applies:

* **Pre-casing** — caliper and video logs were measured ONCE per well,
  before the well was cased. They have no campaign concept. Their
  figures live in ``results/figures/<technique>/`` directly. No
  campaign subfolder.

* **Single-campaign** — SEC techniques (breakpoints, diagnostic,
  sec_robustness, sec_caliper_video) operate on one campaign at a
  time. Their figures live in
  ``results/figures/<technique>/<campaign>/``.

* **Multi-campaign capable** — panels that overlay several campaigns
  in one figure (sec_caliper_panel v11, site_panel v12). Their
  figures live in ``results/figures/<technique>/<campaign>/`` when
  rendering a single campaign and in
  ``results/figures/<technique>/multi_<N>c/`` when overlaying ``N``
  campaigns.

CSVs and other non-figure outputs are NOT relocated by this helper —
they live in technique-specific paths defined elsewhere.

Usage
-----
::

    from karst_analysis.io.figure_paths import resolve_figure_dir

    out_dir = resolve_figure_dir(
        technique_path="convergence/site_panel",
        campaigns=["2022_02", "2022_08"],
    )
    # → Path("results/figures/convergence/site_panel/multi_2c")

    out_dir = resolve_figure_dir(
        technique_path="caliper",
    )
    # → Path("results/figures/caliper")

The convention can be overridden by passing ``output_dir`` to the
caller's CLI; the helper is only invoked when the caller has no
override.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


FIGURES_ROOT = Path("results") / "figures"
"""All figures live below this root directory."""


def resolve_figure_dir(
    technique_path: str,
    *,
    campaigns: Optional[Iterable[str]] = None,
    figures_root: Optional[Path] = None,
) -> Path:
    """Return the canonical output directory for figures of a technique.

    Parameters
    ----------
    technique_path : str
        Forward-slash path of the technique under ``results/figures/``.
        Examples: ``"caliper"``, ``"breakpoints"``,
        ``"convergence/site_panel"``, ``"convergence/sec_caliper_panel"``,
        ``"convergence/sec_caliper_video"``, ``"convergence/sec_caliper"``.
    campaigns : iterable of str, optional
        Campaign identifier(s) the figure is about.

        * ``None`` (or empty) → no campaign subfolder is appended.
          Use this for pre-casing techniques (caliper, caliper_video).
        * One element → the campaign name is appended as a subfolder.
        * More than one → ``multi_<N>c`` is appended.

    figures_root : Path, optional
        Override the figures root (default ``results/figures``). Useful
        in tests with ``tmp_path``.

    Returns
    -------
    Path
        The directory where figures should be written. The caller is
        responsible for creating it (``mkdir(parents=True, exist_ok=True)``).
    """
    root = Path(figures_root) if figures_root is not None else FIGURES_ROOT
    base = root / Path(technique_path)

    if campaigns is None:
        return base

    camp_list = list(campaigns)
    if not camp_list:
        return base
    if len(camp_list) == 1:
        return base / camp_list[0]
    return base / f"multi_{len(camp_list)}c"


def campaign_subdir_label(campaigns: Optional[Iterable[str]]) -> Optional[str]:
    """Return the campaign-subfolder label that would be used.

    Useful for log messages and tests that want to verify the
    placement choice without constructing the full path.
    """
    if campaigns is None:
        return None
    camp_list = list(campaigns)
    if not camp_list:
        return None
    if len(camp_list) == 1:
        return camp_list[0]
    return f"multi_{len(camp_list)}c"

"""Drilling-record loaders.

Currently exposes the Ardaman 2009 lithology + in-situ-conductivity
transcription.  Future records (other reports, other wells) can be
added as additional loaders alongside ``io.load_ardaman``.
"""

from karst_analysis.drilling.io import load_ardaman, DEFAULT_ARDAMAN_CSV

__all__ = ["load_ardaman", "DEFAULT_ARDAMAN_CSV"]

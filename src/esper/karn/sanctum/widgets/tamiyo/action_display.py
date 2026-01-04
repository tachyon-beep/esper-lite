"""Shared action display constants for TamiyoBrain panels.

Centralizes action colors and abbreviations to prevent drift across widgets.
"""

from __future__ import annotations

# Action colors (must remain consistent across Tamiyo subpanels)
ACTION_COLORS: dict[str, str] = {
    "GERMINATE": "green",
    "SET_ALPHA_TARGET": "cyan",
    "FOSSILIZE": "blue",
    "PRUNE": "red",
    "WAIT": "dim",
    "ADVANCE": "cyan",
}

# 4-char abbreviations for compact tables / labels
ACTION_ABBREVS_4: dict[str, str] = {
    "GERMINATE": "GERM",
    "SET_ALPHA_TARGET": "ALPH",
    "FOSSILIZE": "FOSS",
    "PRUNE": "PRUN",
    "WAIT": "WAIT",
    "ADVANCE": "ADVN",
}

# 1-char abbreviations for dense sequence/barchart displays
ACTION_ABBREVS_1: dict[str, str] = {
    "GERMINATE": "G",
    "SET_ALPHA_TARGET": "A",
    "FOSSILIZE": "F",
    "PRUNE": "P",
    "WAIT": "W",
    "ADVANCE": "V",  # V for adVance (A is taken by SET_ALPHA_TARGET)
}


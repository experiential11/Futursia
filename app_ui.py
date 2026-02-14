"""
UI constants, stylesheet, and formatting helpers for the Futursia desktop app.
"""

from datetime import datetime
from typing import Any, List, Union

import pandas as pd
import pytz

# Timezone and refresh
US_EASTERN_TZ = pytz.timezone("US/Eastern")
LOCKED_FAST_REFRESH_MS = 1000

# Fonts and colors
APP_FONT_FAMILY = "Segoe UI"
APP_MONO_FONT_FAMILY = "Consolas"
GREEN_TEXT = "#127a42"
RED_TEXT = "#ba2e3b"
NEUTRAL_TEXT = "#254463"


def get_stylesheet() -> str:
    """Global stylesheet for a clean, high-contrast desktop UI."""
    return """
    QMainWindow, QWidget {
        background: #f4f7fb;
        color: #162233;
        font-family: 'Segoe UI';
        font-size: 11pt;
    }
    QTabWidget::pane {
        border: 1px solid #cfdaea;
        border-radius: 12px;
        background: #edf2f8;
        top: -1px;
    }
    QTabBar::tab {
        background: #dbe5f2;
        color: #2b3f57;
        padding: 10px 16px;
        min-width: 120px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        margin-right: 4px;
        font-weight: 600;
    }
    QTabBar::tab:selected {
        background: #ffffff;
        color: #102740;
    }
    QLabel#pageTitle {
        font-size: 19pt;
        font-weight: 700;
        color: #0e2948;
        padding-bottom: 2px;
    }
    QLabel#sectionTitle {
        font-size: 12pt;
        font-weight: 700;
        color: #12375d;
    }
    QLabel#mutedText {
        color: #5a6f87;
        font-size: 10.5pt;
    }
    QLabel#statusPill {
        background: #e6f0fa;
        color: #1f4d7a;
        border: 1px solid #c0d5ec;
        border-radius: 10px;
        padding: 4px 9px;
        font-size: 10pt;
    }
    QFrame#card {
        background: #ffffff;
        border: 1px solid #d2deec;
        border-radius: 12px;
    }
    QFrame#cardElevated {
        background: #ffffff;
        border: 1px solid #c7d8ea;
        border-radius: 14px;
    }
    QComboBox, QSpinBox, QTextEdit, QTableWidget, QListWidget {
        background: #ffffff;
        border: 1px solid #c4d4e6;
        border-radius: 8px;
        padding: 5px 7px;
        selection-background-color: #d5e7fb;
        selection-color: #162233;
    }
    QComboBox:focus, QSpinBox:focus, QTextEdit:focus, QTableWidget:focus, QListWidget:focus {
        border: 2px solid #0d6886;
    }
    QPushButton {
        background: #0d6886;
        color: #ffffff;
        border: 0;
        border-radius: 8px;
        padding: 7px 12px;
        min-height: 30px;
        font-weight: 600;
    }
    QPushButton:hover {
        background: #0b5973;
    }
    QPushButton:pressed {
        background: #094a5f;
    }
    QPushButton:focus {
        border: 2px solid #f39b22;
    }
    QHeaderView::section {
        background: #e8eef7;
        color: #1d3550;
        border: 0;
        border-bottom: 1px solid #cad7e9;
        padding: 7px;
        font-weight: 600;
    }
    QTableWidget {
        gridline-color: #e3eaf3;
    }
    QTableCornerButton::section {
        background: #e8eef7;
        border: 0;
        border-bottom: 1px solid #cad7e9;
        border-right: 1px solid #cad7e9;
    }
    """


def now_est() -> datetime:
    """Current time in US/Eastern."""
    return datetime.now(US_EASTERN_TZ)


def to_est_text(value: Any, fallback: str = "Unknown") -> str:
    """Convert datetime-like value to US/Eastern display string."""
    if value is None:
        return fallback
    try:
        if isinstance(value, (int, float)):
            ts = pd.to_datetime(int(value), unit="s", utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.notna(ts):
            return ts.tz_convert(US_EASTERN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        pass
    return str(value) if value is not None else fallback


def coerce_timestamp_utc(values: Union[pd.Series, List[Any]]) -> pd.Series:
    """Parse mixed timestamp inputs into UTC-aware pandas timestamps."""
    series = pd.Series(values)
    if series.empty:
        return pd.to_datetime(series, utc=True, errors="coerce")

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = float(numeric.notna().mean()) if len(series) else 0.0
    if numeric_ratio >= 0.95:
        non_null = numeric.dropna()
        abs_max = float(non_null.abs().max()) if not non_null.empty else 0.0
        if abs_max >= 1e17:
            unit = "ns"
        elif abs_max >= 1e14:
            unit = "us"
        elif abs_max >= 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")

    return pd.to_datetime(series, utc=True, errors="coerce")

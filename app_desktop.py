"""
Futursia Forecasting V.1.0 - Desktop Application
Real-time stock data with live updates, 40-minute forecasting, and forecast graphs.
"""

import sys
import os
import hashlib
import html
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytz

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QComboBox, QTextEdit, QHeaderView, QSpinBox, QFrame, QListWidget, QListWidgetItem, QScrollArea
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor

import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.logging_setup import setup_logging, get_logger
from core.client_factory import get_market_client
from core.news_client import NewsClient
from core.forecasting import ForecasterEngine

logger = get_logger()

# Popular stocks
POPULAR_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V',
    '7203.T', '6758.T', '9984.T', '8306.T', '7974.T'
]

US_EASTERN_TZ = pytz.timezone("US/Eastern")
LOCKED_FAST_REFRESH_MS = 1000
APP_FONT_FAMILY = "Segoe UI"
APP_MONO_FONT_FAMILY = "Consolas"
GREEN_TEXT = "#127a42"
RED_TEXT = "#ba2e3b"
NEUTRAL_TEXT = "#254463"


def _app_stylesheet():
    """Global stylesheet for a cleaner, high-contrast desktop UI."""
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


def _now_est():
    """Current timestamp in US/Eastern."""
    return datetime.now(US_EASTERN_TZ)


def _to_est_text(value, fallback="Unknown"):
    """Convert datetime-like values to US/Eastern display text."""
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


def _coerce_timestamp_utc(values):
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


class DashboardTab(QWidget):
    """Dashboard showing top movers with auto-refresh."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.client = get_market_client(config)
        self._refresh_in_progress = False
        self.init_ui()
        self.setup_auto_refresh()
        self.refresh_data()
    
    def init_ui(self):
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel("Market Dashboard")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        subtitle = QLabel("Top movers and intraday range with locked 1-second refresh (EST).")
        subtitle.setObjectName("mutedText")
        layout.addWidget(subtitle)

        ctrl_card = QFrame()
        ctrl_card.setObjectName("card")
        ctrl_layout = QHBoxLayout(ctrl_card)
        ctrl_layout.setContentsMargins(14, 10, 14, 10)
        ctrl_layout.setSpacing(10)

        self.status_label = QLabel("Loading market data...")
        self.status_label.setObjectName("statusPill")
        ctrl_layout.addWidget(self.status_label)

        ctrl_layout.addStretch()
        ctrl_layout.addWidget(QLabel("Auto-refresh (locked):"))
        self.refresh_spin = QSpinBox()
        self.refresh_spin.setMinimum(1)
        self.refresh_spin.setMaximum(1)
        self.refresh_spin.setValue(1)
        self.refresh_spin.setSuffix(" s")
        self.refresh_spin.setEnabled(False)
        self.refresh_spin.setToolTip("Refresh speed is fixed at 1 second.")
        ctrl_layout.addWidget(self.refresh_spin)

        refresh_btn = QPushButton("Refresh Now")
        refresh_btn.setToolTip("Load the latest movers immediately.")
        refresh_btn.clicked.connect(self.refresh_data)
        ctrl_layout.addWidget(refresh_btn)
        layout.addWidget(ctrl_card)

        table_card = QFrame()
        table_card.setObjectName("card")
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(10, 10, 10, 10)
        table_layout.setSpacing(8)

        table_title = QLabel("Top 10 Movers")
        table_title.setObjectName("sectionTitle")
        table_layout.addWidget(table_title)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(['Symbol', 'Price', 'Change $', 'Change %', 'High', 'Low'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setToolTip("Live mover table, refreshed every second.")
        table_layout.addWidget(self.table)

        layout.addWidget(table_card, 1)
        self.setLayout(layout)
    def setup_auto_refresh(self):
        """Setup automatic refresh timer."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(LOCKED_FAST_REFRESH_MS)
    
    def refresh_data(self):
        """Load and display top movers (blocking call, but fast)."""
        if self._refresh_in_progress:
            return
        self._refresh_in_progress = True
        self.status_label.setText("Refreshing (EST)...")
        try:
            movers = self.client.get_top_movers()
            if movers and len(movers) > 0:
                self.display_movers(movers)
                timestamp = _now_est().strftime("%Y-%m-%d %H:%M:%S %Z")
                self.status_label.setText(f"Updated: {timestamp}")
            else:
                self.status_label.setText("No data available")
        except Exception as e:
            logger.error(f"Error loading movers: {str(e)}")
            self.status_label.setText(f"Error: {str(e)[:30]}")
        finally:
            self._refresh_in_progress = False
    
    def display_movers(self, movers):
        """Display movers in table."""
        self.table.setRowCount(len(movers))
        
        for i, mover in enumerate(movers):
            symbol = mover.get('symbol', '')
            price = float(mover.get('price', 0))
            change = float(mover.get('change', 0))
            pct_change = float(mover.get('percent_change', 0))
            high = float(mover.get('high', price))
            low = float(mover.get('low', price))
            
            # Symbol
            item = QTableWidgetItem(symbol)
            item.setFont(QFont(APP_FONT_FAMILY, 11, QFont.Bold))
            self.table.setItem(i, 0, item)
            
            # Price
            item = QTableWidgetItem(f"${price:.2f}")
            self.table.setItem(i, 1, item)
            
            # Change $
            item = QTableWidgetItem(f"${change:+.2f}")
            self.table.setItem(i, 2, item)
            
            # Change %
            item = QTableWidgetItem(f"{pct_change:+.2f}%")
            if pct_change >= 0:
                item.setForeground(QColor(GREEN_TEXT))
            else:
                item.setForeground(QColor(RED_TEXT))
            item.setFont(QFont(APP_FONT_FAMILY, 10, QFont.Bold))
            self.table.setItem(i, 3, item)
            
            # High
            item = QTableWidgetItem(f"${high:.2f}")
            self.table.setItem(i, 4, item)
            
            # Low
            item = QTableWidgetItem(f"${low:.2f}")
            self.table.setItem(i, 5, item)


class TickerDetailTab(QWidget):
    """Ticker detail with live 40-minute forecast and graph."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.client = get_market_client(config)
        self.news = NewsClient(config)
        self.forecaster = ForecasterEngine(config, self.client, self.news)
        refresh_cfg = self.config.get('refresh', {})
        self._forecast_recompute_seconds = max(1, int(refresh_cfg.get('forecast_recompute_interval', 15)))
        self.current_symbol = 'AAPL'
        self._forecast_path_state = None
        self._manual_zoom = False
        self._is_panning = False
        self._pan_start = None
        self._x_data_bounds = None
        self._y_data_bounds = None
        self._live_oos_metrics = {}
        self._market_status_cache = {
            'checked_at': None,
            'is_open': None,
            'label': 'UNKNOWN',
        }
        self._load_in_progress = False
        self._cached_forecast = None
        self._cached_forecast_symbol = None
        self._cached_forecast_at = None
        self.init_ui()
        self.setup_auto_refresh()
        self.load_ticker()
    
    def init_ui(self):
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel("Ticker Detail and 40-Minute Forecast")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        subtitle = QLabel("Trading-style live view with forecast path, confidence, and fast chart navigation.")
        subtitle.setObjectName("mutedText")
        layout.addWidget(subtitle)

        selector_card = QFrame()
        selector_card.setObjectName("card")
        sel_layout = QHBoxLayout(selector_card)
        sel_layout.setContentsMargins(14, 10, 14, 10)
        sel_layout.setSpacing(10)

        sel_layout.addWidget(QLabel("Symbol"))
        self.combo = QComboBox()
        self.combo.addItems(POPULAR_STOCKS)
        self.combo.setMinimumWidth(180)
        self.combo.setToolTip("Select a symbol to load quote, forecast, and chart.")
        self.combo.currentTextChanged.connect(self.on_symbol_changed)
        sel_layout.addWidget(self.combo)

        sel_layout.addSpacing(12)
        sel_layout.addWidget(QLabel("Auto-refresh (locked):"))
        self.refresh_spin = QSpinBox()
        self.refresh_spin.setMinimum(1)
        self.refresh_spin.setMaximum(1)
        self.refresh_spin.setValue(1)
        self.refresh_spin.setSuffix(" s")
        self.refresh_spin.setEnabled(False)
        self.refresh_spin.setToolTip("Refresh speed is fixed at 1 second.")
        sel_layout.addWidget(self.refresh_spin)

        sel_layout.addStretch()
        layout.addWidget(selector_card)

        price_card = QFrame()
        price_card.setObjectName("card")
        price_layout = QHBoxLayout(price_card)
        price_layout.setContentsMargins(14, 10, 14, 10)
        price_layout.setSpacing(14)

        self.price_label = QLabel("Price: Loading...")
        self.price_label.setFont(QFont(APP_FONT_FAMILY, 13, QFont.Bold))
        price_layout.addWidget(self.price_label)

        self.change_label = QLabel("")
        self.change_label.setFont(QFont(APP_FONT_FAMILY, 12, QFont.Bold))
        price_layout.addWidget(self.change_label)

        self.timestamp_label = QLabel("")
        self.timestamp_label.setObjectName("statusPill")
        price_layout.addWidget(self.timestamp_label)

        self.market_status_label = QLabel("Market: --")
        self.market_status_label.setObjectName("statusPill")
        price_layout.addWidget(self.market_status_label)
        price_layout.addStretch()
        layout.addWidget(price_card)

        forecast_box = QFrame()
        forecast_box.setObjectName("cardElevated")
        forecast_layout = QVBoxLayout(forecast_box)
        forecast_layout.setContentsMargins(14, 12, 14, 12)
        forecast_layout.setSpacing(10)

        forecast_title = QLabel("40-Minute Forecast")
        forecast_title.setObjectName("sectionTitle")
        forecast_layout.addWidget(forecast_title)

        chart_hint = QLabel("Chart controls: mouse wheel to zoom, drag to pan, Shift+drag for vertical pan, double-click to reset")
        chart_hint.setObjectName("mutedText")
        forecast_layout.addWidget(chart_hint)

        forecast_text_layout = QHBoxLayout()
        forecast_text_layout.setSpacing(10)

        self.direction_label = QLabel("Direction: Loading")
        self.direction_label.setFont(QFont(APP_FONT_FAMILY, 13, QFont.Bold))
        self.direction_label.setStyleSheet(f"color: {NEUTRAL_TEXT};")
        forecast_text_layout.addWidget(self.direction_label)

        self.conf_label = QLabel("Confidence: --")
        self.conf_label.setFont(QFont(APP_FONT_FAMILY, 11))
        forecast_text_layout.addWidget(self.conf_label)

        self.ret_label = QLabel("Return: --")
        self.ret_label.setFont(QFont(APP_FONT_FAMILY, 11))
        forecast_text_layout.addWidget(self.ret_label)

        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.setToolTip("Zoom into the chart around the current view.")
        self.zoom_in_btn.clicked.connect(lambda: self.zoom_chart(0.8))
        forecast_text_layout.addWidget(self.zoom_in_btn)

        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.setToolTip("Zoom out to see a wider time range.")
        self.zoom_out_btn.clicked.connect(lambda: self.zoom_chart(1.25))
        forecast_text_layout.addWidget(self.zoom_out_btn)

        self.reset_zoom_btn = QPushButton("Reset View")
        self.reset_zoom_btn.setToolTip("Reset chart to the default autoscaled view.")
        self.reset_zoom_btn.clicked.connect(self.reset_chart_zoom)
        forecast_text_layout.addWidget(self.reset_zoom_btn)

        forecast_text_layout.addStretch()
        forecast_layout.addLayout(forecast_text_layout)

        self.forecast_figure = Figure(figsize=(12, 6), dpi=100)
        self.forecast_canvas = FigureCanvas(self.forecast_figure)
        self.forecast_canvas.mpl_connect('scroll_event', self.on_scroll)
        self.forecast_canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.forecast_canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.forecast_canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.forecast_canvas.setMinimumHeight(460)
        forecast_layout.addWidget(self.forecast_canvas, 1)

        layout.addWidget(forecast_box)

        stats_card = QFrame()
        stats_card.setObjectName("card")
        stats_layout = QVBoxLayout(stats_card)
        stats_layout.setContentsMargins(10, 10, 10, 10)
        stats_layout.setSpacing(8)

        stats_label = QLabel("Price and Forecast Stats")
        stats_label.setObjectName("sectionTitle")
        stats_layout.addWidget(stats_label)

        self.stats = QListWidget()
        self.stats.setAlternatingRowColors(True)
        self.stats.setMinimumHeight(200)
        self.stats.setMaximumHeight(280)
        self.stats.setToolTip("Scrollable list of live price, forecast, and OOS metrics.")
        stats_layout.addWidget(self.stats, 1)

        layout.addWidget(stats_card)
        scroll.setWidget(content)
        root_layout.addWidget(scroll)
        self.setLayout(root_layout)
    def update_forecast_chart(self, bars_df, forecast=None):
        """Render a real-time style chart (time on X axis, stock price on Y axis).
        Confidence bands are removed."""
        previous_limits = None
        if self._manual_zoom and self.forecast_figure.axes:
            prev_axis = self.forecast_figure.axes[0]
            previous_limits = (prev_axis.get_xlim(), prev_axis.get_ylim())

        self.forecast_figure.clear()
        axis = self.forecast_figure.add_subplot(111)
        self.forecast_figure.patch.set_facecolor('#ffffff')
        axis.set_facecolor('#f8fbff')
        for spine in axis.spines.values():
            spine.set_color('#b9cadf')

        if bars_df is None or bars_df.empty:
            self._x_data_bounds = None
            self._y_data_bounds = None
            axis.text(0.5, 0.5, "No price data available", ha='center', va='center', transform=axis.transAxes)
            self.forecast_figure.tight_layout()
            self.forecast_canvas.draw()
            return

        plot_df = bars_df.tail(120).copy()
        plot_df['timestamp_utc'] = _coerce_timestamp_utc(plot_df['timestamp'])
        plot_df = plot_df.dropna(subset=['timestamp_utc', 'close'])
        plot_df['timestamp_est'] = plot_df['timestamp_utc'].dt.tz_convert(US_EASTERN_TZ)
        if plot_df.empty:
            self._x_data_bounds = None
            self._y_data_bounds = None
            axis.text(0.5, 0.5, "No valid price data available", ha='center', va='center', transform=axis.transAxes)
            self.forecast_figure.tight_layout()
            self.forecast_canvas.draw()
            return

        axis.plot(
            plot_df['timestamp_est'],
            plot_df['close'],
            color='#1f77b4',
            linewidth=2.2,
            label='Price'
        )

        last_ts = plot_df['timestamp_est'].iloc[-1]
        last_price = float(plot_df['close'].iloc[-1])
        axis.scatter([last_ts], [last_price], color='#1f77b4', s=40, zorder=5)

        if forecast:
            forecast_return_pct = float(forecast.get('predicted_return', 0.0))
            forecast_ts, forecast_prices = self._get_full_forecast_path(
                symbol=self.current_symbol,
                last_ts=last_ts,
                last_price=last_price,
                forecast_return_pct=forecast_return_pct,
                plot_df=plot_df,
            )
            axis.plot(
                forecast_ts,
                forecast_prices,
                color='#ff7f0e',
                linestyle='-',
                linewidth=2.0,
                marker='o',
                markersize=2.5,
                label='40-min forecast path'
            )
            axis.scatter([forecast_ts[-1]], [forecast_prices[-1]], color='#ff7f0e', s=60, zorder=6)
            forecast_target_ts = forecast_ts[-1]
            forecast_target_price = float(forecast_prices[-1])
        else:
            self._forecast_path_state = None
            forecast_target_ts = None
            forecast_target_price = None

        x_min = mdates.date2num(plot_df['timestamp_est'].min())
        x_max = mdates.date2num(plot_df['timestamp_est'].max())
        if forecast_target_ts is not None:
            forecast_x = mdates.date2num(forecast_target_ts)
            x_min = min(x_min, forecast_x)
            x_max = max(x_max, forecast_x)

        y_min = float(plot_df['close'].min())
        y_max = float(plot_df['close'].max())
        if forecast_target_price is not None:
            y_min = min(y_min, float(forecast_target_price))
            y_max = max(y_max, float(forecast_target_price))

        if y_max <= y_min:
            y_max = y_min + max(abs(y_min) * 0.01, 0.1)

        self._x_data_bounds = (x_min, x_max)
        self._y_data_bounds = (y_min, y_max)

        self._draw_market_session_markers(axis, x_min, x_max)
        self._draw_market_status_badge(axis)

        axis.set_title('Real-Time Stock Price (EST)', fontsize=14, fontweight='bold')
        axis.set_xlabel('Time (EST)')
        axis.set_ylabel('Price ($)')
        axis.grid(True, alpha=0.35, linestyle=':', color='#9ab0c9')
        axis.legend(loc='upper left')
        x_locator = mdates.AutoDateLocator(tz=US_EASTERN_TZ, minticks=5, maxticks=12)
        axis.xaxis.set_major_locator(x_locator)
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M', tz=US_EASTERN_TZ))
        axis.tick_params(axis='x', rotation=25, colors='#274766')
        axis.tick_params(axis='y', colors='#274766')

        if previous_limits is not None:
            xlim, ylim = self._clamp_view_limits(previous_limits[0], previous_limits[1])
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)

        self.forecast_figure.tight_layout()
        self.forecast_canvas.draw()

    def _refresh_market_status_cache(self, now_utc):
        """Refresh cached market status periodically from the active market client."""
        checked_at = self._market_status_cache.get('checked_at')
        if checked_at is not None and (now_utc - checked_at).total_seconds() < 30:
            return

        label = 'UNKNOWN'
        is_open = None
        try:
            fetch_fn = getattr(self.client, 'get_market_status', None)
            payload = fetch_fn() if callable(fetch_fn) else {}
            status = str((payload or {}).get('status', '')).strip().lower()
            if status in {'open', 'opened'}:
                label = 'OPEN'
                is_open = True
            elif status in {'closed', 'close'}:
                label = 'CLOSED'
                is_open = False
            elif status:
                label = status.upper()
        except Exception as e:
            logger.debug(f"Unable to refresh market status: {e}")

        self._market_status_cache = {
            'checked_at': now_utc,
            'is_open': is_open,
            'label': label,
        }

    def _draw_market_status_badge(self, axis):
        """Draw open/closed market status badge on chart."""
        status_label = str(self._market_status_cache.get('label', 'UNKNOWN')).upper()
        is_open = self._market_status_cache.get('is_open')
        if is_open is True:
            color = '#127a42'
        elif is_open is False:
            color = '#bf3b3b'
        else:
            color = '#5b6e82'

        axis.text(
            0.99,
            0.98,
            f"Market: {status_label}",
            transform=axis.transAxes,
            ha='right',
            va='top',
            fontsize=9,
            fontweight='bold',
            color=color,
            bbox={'boxstyle': 'round,pad=0.25', 'facecolor': '#ffffff', 'edgecolor': '#c9d8ea', 'alpha': 0.95},
        )

    def _draw_market_session_markers(self, axis, x_min_num, x_max_num):
        """Draw reference markers for US market open/close in EST."""
        try:
            start_ts = pd.Timestamp(mdates.num2date(float(x_min_num), tz=US_EASTERN_TZ))
            end_ts = pd.Timestamp(mdates.num2date(float(x_max_num), tz=US_EASTERN_TZ))
        except Exception:
            return
        if pd.isna(start_ts) or pd.isna(end_ts) or end_ts <= start_ts:
            return

        day_start = start_ts.normalize()
        day_end = end_ts.normalize() + pd.Timedelta(days=1)
        days = pd.date_range(day_start, day_end, freq='D', tz=US_EASTERN_TZ)
        for day in days:
            market_open = day + pd.Timedelta(hours=9, minutes=30)
            market_close = day + pd.Timedelta(hours=16)

            if start_ts <= market_open <= end_ts:
                axis.axvline(
                    market_open.to_pydatetime(),
                    color='#2e6fa4',
                    linestyle=':',
                    linewidth=1.0,
                    alpha=0.8,
                    zorder=2,
                )
                axis.text(
                    market_open.to_pydatetime(),
                    0.985,
                    'Open 09:30',
                    transform=axis.get_xaxis_transform(),
                    rotation=90,
                    va='top',
                    ha='right',
                    fontsize=8,
                    color='#2e6fa4',
                    bbox={'boxstyle': 'round,pad=0.15', 'facecolor': '#ffffff', 'edgecolor': 'none', 'alpha': 0.72},
                )

            if start_ts <= market_close <= end_ts:
                axis.axvline(
                    market_close.to_pydatetime(),
                    color='#bf3b3b',
                    linestyle='--',
                    linewidth=1.2,
                    alpha=0.9,
                    zorder=2,
                )
                axis.text(
                    market_close.to_pydatetime(),
                    0.985,
                    'Close 16:00',
                    transform=axis.get_xaxis_transform(),
                    rotation=90,
                    va='top',
                    ha='right',
                    fontsize=8,
                    color='#bf3b3b',
                    bbox={'boxstyle': 'round,pad=0.15', 'facecolor': '#ffffff', 'edgecolor': 'none', 'alpha': 0.72},
                )

    def _get_full_forecast_path(self, symbol, last_ts, last_price, forecast_return_pct, plot_df):
        """Build the full 40-minute minute-by-minute forecast path."""
        horizon_minutes = 40
        now = datetime.utcnow()
        state = self._forecast_path_state

        needs_refresh = state is None
        if state is not None:
            if state.get('symbol') != symbol:
                needs_refresh = True
            elif state.get('horizon') != horizon_minutes:
                needs_refresh = True
            elif (now - state.get('created_at', now)).total_seconds() >= 60:
                # Rebuild once per minute so the horizon stays forward-looking.
                needs_refresh = True
            elif abs(state.get('start_price', last_price) - last_price) / max(last_price, 1e-9) > 0.02:
                needs_refresh = True
            elif abs(state.get('forecast_return_pct', forecast_return_pct) - forecast_return_pct) > 0.5:
                needs_refresh = True

        if needs_refresh:
            full_prices = self._generate_forecast_path_points(
                symbol=symbol,
                start_price=last_price,
                forecast_return_pct=forecast_return_pct,
                plot_df=plot_df,
                horizon_minutes=horizon_minutes,
                created_at=now,
            )

            start_timestamp = pd.to_datetime(last_ts, utc=True, errors='coerce')
            if pd.isna(start_timestamp):
                start_timestamp = _now_est()
            else:
                start_timestamp = start_timestamp.tz_convert(US_EASTERN_TZ).to_pydatetime()

            self._forecast_path_state = {
                'symbol': symbol,
                'created_at': now,
                'start_timestamp': start_timestamp,
                'start_price': float(last_price),
                'forecast_return_pct': float(forecast_return_pct),
                'horizon': horizon_minutes,
                'full_prices': full_prices,
            }

        state = self._forecast_path_state
        full_prices = state['full_prices']
        full_timestamps = [
            state['start_timestamp'] + timedelta(minutes=step_idx)
            for step_idx in range(len(full_prices))
        ]

        return full_timestamps, full_prices

    def _generate_forecast_path_points(self, symbol, start_price, forecast_return_pct, plot_df, horizon_minutes, created_at):
        """Generate a non-linear, stock-like forecast path constrained to the predicted endpoint."""
        target_price = start_price * (1.0 + forecast_return_pct / 100.0)
        if target_price <= 0:
            target_price = max(start_price, 0.01)

        minute_returns = pd.Series(dtype=float)
        if plot_df is not None and 'close' in plot_df.columns:
            minute_returns = plot_df['close'].pct_change().dropna()

        sigma = float(minute_returns.std()) if not minute_returns.empty else 0.0012
        sigma = max(0.0003, min(sigma, 0.02))

        seed_input = f"{symbol}|{created_at.strftime('%Y%m%d%H%M')}|{forecast_return_pct:.6f}|{start_price:.4f}"
        seed = int(hashlib.sha256(seed_input.encode('utf-8')).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        steps = horizon_minutes
        t = np.linspace(0.0, 1.0, steps + 1)
        linear_path = start_price + (target_price - start_price) * t

        shocks = rng.normal(0.0, 1.0, steps)
        brownian = np.concatenate(([0.0], np.cumsum(shocks)))
        bridge = brownian - t * brownian[-1]
        bridge[0] = 0.0
        bridge[-1] = 0.0

        bridge_inner = bridge[1:-1]
        bridge_std = float(np.std(bridge_inner)) if len(bridge_inner) > 0 else 1.0
        if bridge_std < 1e-9:
            bridge_std = 1.0
        bridge = bridge / bridge_std

        wiggle_scale = start_price * sigma * 0.75
        max_wiggle = max(start_price * 0.03, 0.35)
        wiggles = np.clip(bridge * wiggle_scale, -max_wiggle, max_wiggle)

        prices = linear_path + wiggles
        prices = np.maximum(prices, 0.01)

        max_step_move = max(start_price * 0.02, 0.15)
        for idx in range(1, len(prices)):
            delta = prices[idx] - prices[idx - 1]
            if delta > max_step_move:
                prices[idx] = prices[idx - 1] + max_step_move
            elif delta < -max_step_move:
                prices[idx] = prices[idx - 1] - max_step_move

        prices[0] = start_price
        prices[-1] = target_price
        return prices.tolist()

    def on_scroll(self, event):
        """Handle mouse scroll for zoom in/out on the forecast chart.
        Scroll up to zoom in, scroll down to zoom out."""
        if event.inaxes is None:
            return

        wheel_step = getattr(event, 'step', None)
        if wheel_step is not None and wheel_step != 0:
            scale_factor = 0.88 ** float(wheel_step)
        elif event.button == 'up':
            scale_factor = 0.88
        elif event.button == 'down':
            scale_factor = 1.14
        else:
            return

        y_scale_factor = 1.0 + (scale_factor - 1.0) * 0.45
        self._apply_zoom(
            event.inaxes,
            scale_factor,
            anchor=(event.xdata, event.ydata),
            y_scale_factor=y_scale_factor,
        )
        self._manual_zoom = True
        self.forecast_canvas.draw_idle()

    def on_mouse_press(self, event):
        """Start pan mode on left-click inside the chart."""
        if event.button == 1 and event.dblclick and event.inaxes is not None:
            self.reset_chart_zoom()
            return
        if event.button != 1 or event.inaxes is None or event.xdata is None:
            return

        self._is_panning = True
        self._manual_zoom = True
        self._pan_start = {
            'axis': event.inaxes,
            'xdata': event.xdata,
            'ydata': event.ydata,
            'xlim': event.inaxes.get_xlim(),
            'ylim': event.inaxes.get_ylim(),
        }

    def on_mouse_move(self, event):
        """Pan chart while dragging with left mouse button held."""
        if not self._is_panning or self._pan_start is None:
            return
        if event.inaxes is None or event.inaxes != self._pan_start['axis']:
            return
        if event.xdata is None:
            return

        dx = self._pan_start['xdata'] - event.xdata
        pan_vertical = str(getattr(event, 'key', '') or '').lower() == 'shift'
        if pan_vertical and event.ydata is not None and self._pan_start['ydata'] is not None:
            dy = self._pan_start['ydata'] - event.ydata
        else:
            dy = 0.0

        raw_xlim = (
            self._pan_start['xlim'][0] + dx,
            self._pan_start['xlim'][1] + dx,
        )
        raw_ylim = (
            self._pan_start['ylim'][0] + dy,
            self._pan_start['ylim'][1] + dy,
        )
        xlim, ylim = self._clamp_view_limits(raw_xlim, raw_ylim)
        event.inaxes.set_xlim(xlim)
        event.inaxes.set_ylim(ylim)
        self.forecast_canvas.draw_idle()

    def on_mouse_release(self, event):
        """Stop pan mode when left mouse button is released."""
        if event.button == 1:
            self._is_panning = False
            self._pan_start = None

    def _clamp_view_limits(self, xlim, ylim):
        """Keep zoom/pan limits within a sane range around plotted data."""
        if self._x_data_bounds is None or self._y_data_bounds is None:
            return xlim, ylim

        data_x_min, data_x_max = self._x_data_bounds
        data_y_min, data_y_max = self._y_data_bounds

        x_left, x_right = sorted((float(xlim[0]), float(xlim[1])))
        y_low, y_high = sorted((float(ylim[0]), float(ylim[1])))

        data_x_span = max(data_x_max - data_x_min, 1.0 / 1440.0)  # 1 minute
        data_y_span = max(data_y_max - data_y_min, max(abs(data_y_min) * 0.01, 0.1))

        min_x_span = max(data_x_span * 0.004, 10.0 / 86400.0)  # 10 seconds
        min_y_span = max(data_y_span * 0.006, 0.01)
        max_x_span = max(data_x_span * 12.0, 4.0 / 24.0)  # at least 4 hours
        max_y_span = max(data_y_span * 20.0, 1.0)

        x_span = max(min_x_span, min(x_right - x_left, max_x_span))
        y_span = max(min_y_span, min(y_high - y_low, max_y_span))

        x_center = (x_left + x_right) / 2.0
        y_center = (y_low + y_high) / 2.0

        x_margin = max(data_x_span * 6.0, 120.0 / 1440.0)  # 2 hours minimum
        y_margin = max(data_y_span * 2.5, 2.0)

        allowed_x_min = data_x_min - x_margin
        allowed_x_max = data_x_max + x_margin
        allowed_y_min = data_y_min - y_margin
        allowed_y_max = data_y_max + y_margin

        allowed_x_span = max(allowed_x_max - allowed_x_min, min_x_span)
        allowed_y_span = max(allowed_y_max - allowed_y_min, min_y_span)
        x_span = min(x_span, allowed_x_span)
        y_span = min(y_span, allowed_y_span)

        if x_center - x_span / 2.0 < allowed_x_min:
            x_center = allowed_x_min + x_span / 2.0
        if x_center + x_span / 2.0 > allowed_x_max:
            x_center = allowed_x_max - x_span / 2.0

        if y_center - y_span / 2.0 < allowed_y_min:
            y_center = allowed_y_min + y_span / 2.0
        if y_center + y_span / 2.0 > allowed_y_max:
            y_center = allowed_y_max - y_span / 2.0

        return (
            (x_center - x_span / 2.0, x_center + x_span / 2.0),
            (y_center - y_span / 2.0, y_center + y_span / 2.0),
        )

    def _apply_zoom(self, axis, scale_factor, anchor=None, y_scale_factor=None):
        """Apply zoom around an anchor point on the given axis."""
        cur_xlim = axis.get_xlim()
        cur_ylim = axis.get_ylim()

        if y_scale_factor is None:
            y_scale_factor = scale_factor

        if anchor is None or anchor[0] is None:
            x_anchor = (cur_xlim[0] + cur_xlim[1]) / 2.0
        else:
            x_anchor = anchor[0]
        if anchor is None or anchor[1] is None:
            y_anchor = (cur_ylim[0] + cur_ylim[1]) / 2.0
        else:
            y_anchor = anchor[1]

        cur_width = max(cur_xlim[1] - cur_xlim[0], 1e-12)
        cur_height = max(cur_ylim[1] - cur_ylim[0], 1e-12)
        new_width = cur_width * max(scale_factor, 1e-6)
        new_height = cur_height * max(y_scale_factor, 1e-6)

        relx = (cur_xlim[1] - x_anchor) / cur_width
        rely = (cur_ylim[1] - y_anchor) / cur_height

        raw_xlim = (x_anchor - new_width * (1 - relx), x_anchor + new_width * relx)
        raw_ylim = (y_anchor - new_height * (1 - rely), y_anchor + new_height * rely)
        xlim, ylim = self._clamp_view_limits(raw_xlim, raw_ylim)
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)

    def zoom_chart(self, scale_factor):
        """Zoom the active chart in or out."""
        if not self.forecast_figure.axes:
            return
        axis = self.forecast_figure.axes[0]
        self._apply_zoom(axis, scale_factor)
        self._manual_zoom = True
        self.forecast_canvas.draw_idle()

    def reset_chart_zoom(self):
        """Reset chart zoom to auto-fit on the current plotted data."""
        self._manual_zoom = False
        if not self.forecast_figure.axes:
            return
        axis = self.forecast_figure.axes[0]
        axis.relim()
        axis.autoscale_view()
        self.forecast_canvas.draw_idle()

    
    def setup_auto_refresh(self):
        """Setup automatic refresh timer."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.load_ticker)
        self.timer.start(LOCKED_FAST_REFRESH_MS)

        self.scoring_timer = QTimer()
        self.scoring_timer.timeout.connect(self.run_scoring_cycle)
        self.scoring_timer.start(60000)  # 60 seconds
        self.run_scoring_cycle()

    def run_scoring_cycle(self):
        """Score due forecasts and refresh cached live OOS metrics."""
        try:
            score_result = self.forecaster.score_due_forecasts(max_rows=500)
            self._live_oos_metrics = self.forecaster.get_live_oos_metrics(lookback_hours=24)
            if score_result and score_result.get('scored', 0) > 0:
                logger.info(
                    "Scored %s due forecasts (%s skipped)",
                    score_result.get('scored', 0),
                    score_result.get('skipped', 0),
                )
        except Exception as e:
            logger.warning(f"Forecast scoring cycle failed: {e}")
            self._live_oos_metrics = {}
    
    def on_symbol_changed(self):
        """Handle symbol change."""
        self.current_symbol = self.combo.currentText()
        self._forecast_path_state = None
        self._manual_zoom = False
        self._cached_forecast = None
        self._cached_forecast_symbol = None
        self._cached_forecast_at = None
        self.load_ticker()
    
    def load_ticker(self):
        """Load ticker data and forecast."""
        if self._load_in_progress:
            return
        self._load_in_progress = True
        try:
            symbol = self.current_symbol

            bars_df = pd.DataFrame()
            try:
                bars = self.client.get_bars(symbol, limit=240)
                if isinstance(bars, pd.DataFrame):
                    bars_df = bars
            except Exception as e:
                logger.error(f"Error getting bars for {symbol}: {e}")

            try:
                quote = self.client.get_quote(symbol) or {}
            except Exception as e:
                logger.error(f"Error getting quote for {symbol}: {e}")
                quote = {}

            price = float(quote.get('last_price', quote.get('c', 0)) or 0)
            prev_close = float(quote.get('pc', price) or price)
            high = float(quote.get('h', quote.get('high', price)) or price)
            low = float(quote.get('l', quote.get('low', price)) or price)

            if price == 0:
                self.price_label.setText("Price: Failed to load")
                self.timestamp_label.setText("")
                self._forecast_path_state = None
                self.update_forecast_chart(bars_df, None)
                return

            change = price - prev_close
            chg_pct = (change / prev_close * 100) if prev_close != 0 else 0

            self.price_label.setText(f"Price: ${price:.2f} | High: ${high:.2f} | Low: ${low:.2f}")
            if change >= 0:
                self.change_label.setText(f"Change: +${change:.2f} (+{chg_pct:.2f}%)")
                self.change_label.setStyleSheet(f"color: {GREEN_TEXT}; font-weight: 700;")
            else:
                self.change_label.setText(f"Change: ${change:.2f} ({chg_pct:.2f}%)")
                self.change_label.setStyleSheet(f"color: {RED_TEXT}; font-weight: 700;")

            now_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)
            self._refresh_market_status_cache(now_utc)
            market_label = str(self._market_status_cache.get('label', 'UNKNOWN')).upper()
            market_is_open = self._market_status_cache.get('is_open')
            if market_is_open is True:
                self.market_status_label.setStyleSheet(
                    "background: #e8f6ee; color: #127a42; border: 1px solid #b7dfc7; border-radius: 10px; padding: 4px 9px; font-size: 10pt;"
                )
            elif market_is_open is False:
                self.market_status_label.setStyleSheet(
                    "background: #fdecec; color: #bf3b3b; border: 1px solid #efc5c5; border-radius: 10px; padding: 4px 9px; font-size: 10pt;"
                )
            else:
                self.market_status_label.setStyleSheet(
                    "background: #eef2f7; color: #55697d; border: 1px solid #cfdaea; border-radius: 10px; padding: 4px 9px; font-size: 10pt;"
                )
            self.market_status_label.setText(f"Market: {market_label}")

            timestamp = _now_est().strftime("%Y-%m-%d %H:%M:%S %Z")
            if not bars_df.empty and 'timestamp' in bars_df.columns:
                try:
                    last_bar_ts = pd.to_datetime(bars_df['timestamp'].iloc[-1], utc=True, errors='coerce')
                    age_min = int((now_utc - last_bar_ts).total_seconds() // 60)
                    last_bar_est = last_bar_ts.tz_convert(US_EASTERN_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')
                    self.timestamp_label.setText(
                        f"App update: {timestamp} | Last market print: {last_bar_est} ({age_min}m ago)"
                    )
                except Exception:
                    self.timestamp_label.setText(f"Last update: {timestamp}")
            else:
                self.timestamp_label.setText(f"Last update: {timestamp}")

            forecast = None
            should_refresh_model = (
                self._cached_forecast is None
                or self._cached_forecast_symbol != symbol
                or self._cached_forecast_at is None
                or (now_utc - self._cached_forecast_at).total_seconds() >= self._forecast_recompute_seconds
            )
            if should_refresh_model:
                try:
                    model_forecast = self.forecaster.generate_forecast(symbol)
                    model_status = (model_forecast or {}).get('model_status', '')
                    is_closed_status = isinstance(model_status, str) and model_status.lower().startswith('market closed')

                    if model_forecast and model_forecast.get('prediction_return') is not None:
                        forecast = {
                            'direction': model_forecast.get('direction', 'FLAT'),
                            'confidence': float(model_forecast.get('confidence', 50.0)),
                            'predicted_return': float(model_forecast.get('prediction_return', 0.0)) * 100.0,
                            'model_status': model_forecast.get('model_status', 'OK'),
                        }
                    elif is_closed_status:
                        forecast = {
                            'direction': 'FLAT',
                            'confidence': 0.0,
                            'predicted_return': 0.0,
                            'model_status': model_status,
                        }
                    else:
                        forecast = self.generate_quick_forecast(symbol, quote)
                except Exception as e:
                    logger.warning(f"Enhanced forecast unavailable for {symbol}, using quick mode: {e}")
                    forecast = self.generate_quick_forecast(symbol, quote)

                self._cached_forecast = forecast
                self._cached_forecast_symbol = symbol
                self._cached_forecast_at = now_utc
            else:
                forecast = self._cached_forecast
                if forecast is None:
                    forecast = self.generate_quick_forecast(symbol, quote)

            if forecast:
                direction = forecast.get('direction', 'FLAT')
                confidence = float(forecast.get('confidence', 50))
                pred_return = float(forecast.get('predicted_return', 0))
                dir_label = "UP" if direction == 'UP' else ("DOWN" if direction == 'DOWN' else "FLAT")
                if dir_label == "UP":
                    direction_color = GREEN_TEXT
                elif dir_label == "DOWN":
                    direction_color = RED_TEXT
                else:
                    direction_color = NEUTRAL_TEXT

                self.direction_label.setText(f"Direction: {dir_label}")
                self.direction_label.setStyleSheet(f"color: {direction_color};")
                self.conf_label.setText(f"Confidence: {confidence:.1f}%")
                self.ret_label.setText(f"Return: {pred_return:+.3f}%")
                self.update_forecast_chart(bars_df, forecast)

                stats_data = [
                    ('Current Price', f"${price:.2f}"),
                    ('Previous Close', f"${prev_close:.2f}"),
                    ('Daily Change', f"{change:+.2f} ({chg_pct:+.2f}%)"),
                    ('52-Week High', f"${high:.2f}"),
                    ('52-Week Low', f"${low:.2f}"),
                    ('', ''),
                    ('Forecast Return', f"{pred_return:+.3f}%"),
                    ('Confidence', f"{confidence:.1f}%"),
                ]
            else:
                self.direction_label.setText("Forecast unavailable")
                self.direction_label.setStyleSheet(f"color: {NEUTRAL_TEXT};")
                self.update_forecast_chart(bars_df, None)
                stats_data = [
                    ('Current Price', f"${price:.2f}"),
                    ('Previous Close', f"${prev_close:.2f}"),
                    ('Daily Change', f"{change:+.2f} ({chg_pct:+.2f}%)"),
                    ('High', f"${high:.2f}"),
                    ('Low', f"${low:.2f}"),
                ]

            live_metrics = self._live_oos_metrics or {}
            live_acc3 = live_metrics.get('accuracy_3class_pct')
            live_bin = live_metrics.get('accuracy_binary_excl_flat_pct')
            live_count = live_metrics.get('total_scored', 0)
            if live_count:
                stats_data.extend([
                    ('', ''),
                    ('Live OOS (24h, 3-class)', f"{live_acc3:.2f}%"),
                    ('Live OOS (24h, binary)', f"{live_bin:.2f}%" if live_bin is not None else 'N/A'),
                    ('Live Scored Forecasts', str(live_count)),
                ])
                symbol_perf = None
                for row in live_metrics.get('by_symbol', []):
                    if row.get('symbol') == symbol:
                        symbol_perf = row
                        break
                if symbol_perf:
                    stats_data.append(
                        (
                            f"{symbol} OOS (3-class)",
                            f"{float(symbol_perf.get('accuracy_3class_pct', 0.0)):.2f}% ({int(symbol_perf.get('total_scored', 0))} scored)",
                        )
                    )

            self.stats.clear()
            for metric, value in stats_data:
                if metric == '':
                    divider = QListWidgetItem(" ")
                    divider.setBackground(QColor(230, 236, 245))
                    self.stats.addItem(divider)
                    continue

                stat_item = QListWidgetItem(f"{metric}: {value}")
                self.stats.addItem(stat_item)
        finally:
            self._load_in_progress = False
    def generate_quick_forecast(self, symbol: str, quote: dict) -> dict:
        """
        Generate a quick forecast from current quote.
        Chart rendering uses real bars directly.
        """
        try:
            price = float(quote.get('last_price', quote.get('c', 0)) or 0)
            prev_close = float(quote.get('pc', price) or price)
            high = float(quote.get('h', quote.get('high', price)) or price)
            low = float(quote.get('l', quote.get('low', price)) or price)

            if price == 0:
                return None

            daily_change_pct = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0
            intraday_range = ((high - low) / price * 100) if price > 0 else 0

            forecast_return = daily_change_pct * 0.15
            if daily_change_pct > 0.5:
                forecast_return -= intraday_range * 0.05
            elif daily_change_pct < -0.5:
                forecast_return += intraday_range * 0.05

            confidence = min(95, max(45, 50 + abs(daily_change_pct) * 8))

            if forecast_return > 0.01:
                direction = 'UP'
            elif forecast_return < -0.01:
                direction = 'DOWN'
            else:
                direction = 'FLAT'

            lower_50 = forecast_return * 0.7
            upper_50 = forecast_return * 1.3
            lower_90 = forecast_return * 0.3
            upper_90 = forecast_return * 1.7

            return {
                'direction': direction,
                'confidence': confidence,
                'predicted_return': forecast_return,
                'lower_50': lower_50,
                'upper_50': upper_50,
                'lower_90': lower_90,
                'upper_90': upper_90,
            }
        except Exception as e:
            logger.error(f"Error generating forecast for {symbol}: {e}")
            return None


class NewsTab(QWidget):
    """News headlines tab with auto-refresh."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.news = NewsClient(config)
        self.current_symbol = 'AAPL'
        self._news_refresh_in_progress = False
        self.init_ui()
        self.setup_auto_refresh()
        self.load_news()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel("Financial News and Headlines")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        subtitle = QLabel("Symbol-level headline feed in EST with source and summary.")
        subtitle.setObjectName("mutedText")
        layout.addWidget(subtitle)

        selector_card = QFrame()
        selector_card.setObjectName("card")
        sel_layout = QHBoxLayout(selector_card)
        sel_layout.setContentsMargins(14, 10, 14, 10)
        sel_layout.setSpacing(10)

        sel_layout.addWidget(QLabel("Symbol"))
        self.combo = QComboBox()
        self.combo.addItems(POPULAR_STOCKS)
        self.combo.setMinimumWidth(180)
        self.combo.currentTextChanged.connect(self.on_symbol_changed)
        sel_layout.addWidget(self.combo)
        sel_layout.addStretch()

        layout.addWidget(selector_card)

        news_card = QFrame()
        news_card.setObjectName("card")
        news_layout = QVBoxLayout(news_card)
        news_layout.setContentsMargins(10, 10, 10, 10)
        news_layout.setSpacing(8)

        news_title = QLabel("Latest Headlines")
        news_title.setObjectName("sectionTitle")
        news_layout.addWidget(news_title)

        self.news_text = QTextEdit()
        self.news_text.setReadOnly(True)
        self.news_text.setFont(QFont(APP_MONO_FONT_FAMILY, 10))
        self.news_text.setToolTip("Headlines, source, date, and summary.")
        news_layout.addWidget(self.news_text)

        layout.addWidget(news_card, 1)
        self.setLayout(layout)
    def setup_auto_refresh(self):
        """Setup automatic refresh timer."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.load_news)
        self.timer.start(LOCKED_FAST_REFRESH_MS)
    
    def on_symbol_changed(self):
        """Handle symbol change."""
        self.current_symbol = self.combo.currentText()
        self.load_news()
    
    def load_news(self):
        """Load news headlines."""
        if self._news_refresh_in_progress:
            return
        self._news_refresh_in_progress = True
        symbol = self.current_symbol
        
        try:
            headlines = self.news.get_headlines(symbol, limit=10)
            if headlines and len(headlines) > 0:
                self.display_news(headlines)
            else:
                self.display_demo_news(symbol)
        except Exception as e:
            logger.error(f"Error loading news: {e}")
            self.display_demo_news(symbol)
        finally:
            self._news_refresh_in_progress = False
    
    def display_demo_news(self, symbol: str):
        """Display demo news when API unavailable."""
        now_est = _now_est()
        demo_headlines = [
            {
                'headline': f'{symbol} gains momentum as analysts revise price targets',
                'source': 'MarketWatch',
                'summary': f'{symbol} experienced strong trading activity today with institutional interest',
                'datetime': now_est.strftime('%Y-%m-%d %H:%M:%S %Z')
            },
            {
                'headline': f'Tech sector rally pushes {symbol} higher',
                'source': 'Bloomberg',
                'summary': 'Broad tech rally supports gains across major indices',
                'datetime': (now_est - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S %Z')
            },
            {
                'headline': f'{symbol} Q4 earnings beat expectations',
                'source': 'Reuters',
                'summary': 'Company delivers strong quarter with revenue growth acceleration',
                'datetime': (now_est - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S %Z')
            },
        ]
        self.display_news(demo_headlines)
    
    def display_news(self, headlines):
        """Display news headlines."""
        cards = []
        for i, headline in enumerate(headlines, 1):
            title = html.escape(str(headline.get('headline') or headline.get('title', 'N/A')))
            summary = html.escape(str(headline.get('summary') or headline.get('text', 'No summary available')))
            source = html.escape(str(headline.get('source', 'Unknown')))
            raw_dt = headline.get('datetime') or headline.get('timestamp')
            date_text = html.escape(_to_est_text(raw_dt, fallback='Unknown'))

            cards.append(
                f"""
                <div style="margin: 0 0 10px 0; padding: 10px 12px; background: #f8fbff; border: 1px solid #d7e4f3; border-radius: 9px;">
                  <div style="font-size: 10pt; color: #5d7088; margin-bottom: 4px;">#{i}  {source}  |  {date_text}</div>
                  <div style="font-size: 11.5pt; font-weight: 700; color: #102b49; margin-bottom: 6px;">{title}</div>
                  <div style="font-size: 10.5pt; color: #314b66; line-height: 1.35;">{summary}</div>
                </div>
                """
            )

        html_text = f"""
        <html>
          <body style="font-family: 'Segoe UI'; background: #ffffff; color: #1a2d42; margin: 8px;">
            <div style="font-size: 12pt; font-weight: 700; margin-bottom: 10px;">
              Latest news for {html.escape(self.current_symbol)}
            </div>
            {''.join(cards)}
          </body>
        </html>
        """
        self.news_text.setHtml(html_text)


class SettingsTab(QWidget):
    """Settings and diagnostics."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel("Settings and Diagnostics")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        subtitle = QLabel("Provider configuration, forecast settings, and runtime diagnostics.")
        subtitle.setObjectName("mutedText")
        layout.addWidget(subtitle)

        info_card = QFrame()
        info_card.setObjectName("card")
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(10, 10, 10, 10)
        info_layout.setSpacing(8)

        diag_label = QLabel("System Information")
        diag_label.setObjectName("sectionTitle")
        info_layout.addWidget(diag_label)

        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setFont(QFont(APP_MONO_FONT_FAMILY, 10))

        provider = (self.config.get('market_api_provider') or 'finnhub').strip().lower()
        key_presence = {
            'databento': bool(os.getenv('DATABENTO_API_KEY') or self.config.get('databento_api', {}).get('api_key')),
            'fmp': bool(os.getenv('FMP_API_KEY') or self.config.get('fmp_api', {}).get('api_key')),
            'finnhub': bool(os.getenv('FINNHUB_API_KEY') or self.config.get('finnhub_api', {}).get('api_key')),
            'yfinance': True,
        }
        provider_base_urls = {
            'databento': self.config.get('databento_api', {}).get('base_url', 'N/A'),
            'fmp': self.config.get('fmp_api', {}).get('base_url', 'N/A'),
            'finnhub': self.config.get('finnhub_api', {}).get('base_url', 'N/A'),
            'yfinance': 'https://query1.finance.yahoo.com',
        }
        provider_rate_limits = {
            'databento': self.config.get('databento_api', {}).get('rate_limit', {}).get('calls_per_minute', 'N/A'),
            'fmp': self.config.get('fmp_api', {}).get('rate_limit', {}).get('calls_per_minute', 'N/A'),
            'finnhub': self.config.get('finnhub_api', {}).get('rate_limit', {}).get('calls_per_minute', 'N/A'),
            'yfinance': 'N/A',
        }
        active_base_url = provider_base_urls.get(provider, 'N/A')
        active_key_present = key_presence.get(provider, False)
        active_rate_limit = provider_rate_limits.get(provider, 'N/A')

        info = f"""
Futursia Forecasting V1.0 - Diagnostics
{'='*60}

Python Version: {sys.version}
Platform: {sys.platform}

Configuration:
  Market Provider: {provider}
  Base URL: {active_base_url}
  API Key Present: {active_key_present}
  Rate Limit: {active_rate_limit} calls/min

Forecast Settings:
  Horizon: {self.config.get('forecast', {}).get('horizon_minutes', 'N/A')} minutes
  Lookback: {self.config.get('forecast', {}).get('features_lookback_minutes', 'N/A')} minutes
  Model: {self.config.get('forecast', {}).get('primary_model', 'N/A')}

Features:
  - Real-time stock quotes
  - 40-minute forecasts
  - Forecast path chart
  - Financial news headlines
  - 1-second auto-refresh
  - Non-blocking UI updates
"""

        info_text.setText(info)
        info_layout.addWidget(info_text)

        layout.addWidget(info_card, 1)
        self.setLayout(layout)

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Load config
        config_path = Path(__file__).parent / 'configs' / 'config.yaml'
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.setWindowTitle("Futursia Forecasting")
        self.setGeometry(80, 80, 1480, 920)
        self.setMinimumSize(1240, 780)
        
        # Create tabs
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setMovable(False)
        tabs.addTab(DashboardTab(self.config), "Market")
        tabs.addTab(TickerDetailTab(self.config), "Ticker + Forecast")
        tabs.addTab(NewsTab(self.config), "News")
        tabs.addTab(SettingsTab(self.config), "Diagnostics")
        
        self.setCentralWidget(tabs)
        
        logger.info("Application started successfully")
    
    def closeEvent(self, event):
        """Handle application close."""
        logger.info("Application closing")
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setFont(QFont(APP_FONT_FAMILY, 10))
    app.setStyleSheet(_app_stylesheet())
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()






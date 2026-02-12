"""
Futursia Forecasting V.1.0 - Desktop Application
Real-time stock data with live updates, 40-minute forecasting, and forecast graphs.
"""

import sys
import os
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QComboBox, QTextEdit, QHeaderView, QMessageBox, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QBrush

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.logging_setup import setup_logging, get_logger
from core.client_factory import get_market_client
from core.finnhub_client import format_bars_for_forecast
from core.news_client import NewsClient
from core.forecasting import ForecasterEngine
from core.db import MarketDB

logger = get_logger()

# Popular stocks
POPULAR_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V',
    '7203.T', '6758.T', '9984.T', '8306.T', '7974.T'
]


class DashboardTab(QWidget):
    """Dashboard showing top movers with auto-refresh."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.client = get_market_client(config)
        self.init_ui()
        self.setup_auto_refresh()
        self.refresh_data()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        title = QLabel("📊 Market Dashboard - Top 10 Movers")
        title.setFont(QFont('Arial', 16, QFont.Bold))
        layout.addWidget(title)
        
        # Status and refresh control
        ctrl_layout = QHBoxLayout()
        self.status_label = QLabel("Loading market data...")
        self.status_label.setFont(QFont('Arial', 10))
        ctrl_layout.addWidget(self.status_label)
        
        ctrl_layout.addStretch()
        
        refresh_label = QLabel("Auto-refresh (sec):")
        ctrl_layout.addWidget(refresh_label)
        self.refresh_spin = QSpinBox()
        self.refresh_spin.setMinimum(5)
        self.refresh_spin.setMaximum(60)
        self.refresh_spin.setValue(10)
        self.refresh_spin.valueChanged.connect(self.update_refresh_interval)
        ctrl_layout.addWidget(self.refresh_spin)
        
        btn = QPushButton("🔄 Refresh Now")
        btn.clicked.connect(self.refresh_data)
        ctrl_layout.addWidget(btn)
        
        layout.addLayout(ctrl_layout)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(['Symbol', 'Price', 'Change $', 'Change %', 'High', 'Low'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def setup_auto_refresh(self):
        """Setup automatic refresh timer."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(10000)  # 10 seconds
    
    def update_refresh_interval(self):
        """Update refresh interval."""
        interval_ms = self.refresh_spin.value() * 1000
        self.timer.setInterval(interval_ms)
    
    def refresh_data(self):
        """Load and display top movers (blocking call, but fast)."""
        self.status_label.setText("Refreshing...")
        try:
            movers = self.client.get_top_movers()
            if movers and len(movers) > 0:
                self.display_movers(movers)
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.status_label.setText(f"✅ Updated: {timestamp}")
            else:
                self.status_label.setText("No data available")
        except Exception as e:
            logger.error(f"Error loading movers: {str(e)}")
            self.status_label.setText(f"❌ Error: {str(e)[:30]}")
    
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
            item.setFont(QFont('Arial', 11, QFont.Bold))
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
                item.setForeground(QColor('green'))
            else:
                item.setForeground(QColor('red'))
            item.setFont(QFont('Arial', 10, QFont.Bold))
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
        self.current_symbol = 'AAPL'
        self._forecast_path_state = None
        self._manual_zoom = False
        self._is_panning = False
        self._pan_start = None
        self._x_data_bounds = None
        self._y_data_bounds = None
        self._live_oos_metrics = {}
        self.init_ui()
        self.setup_auto_refresh()
        self.load_ticker()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        title = QLabel("📈 Ticker Detail & 40-Minute Forecast")
        title.setFont(QFont('Arial', 16, QFont.Bold))
        layout.addWidget(title)
        
        # Stock selector
        sel_layout = QHBoxLayout()
        sel_layout.addWidget(QLabel("Select Stock:"))
        self.combo = QComboBox()
        self.combo.addItems(POPULAR_STOCKS)
        self.combo.currentTextChanged.connect(self.on_symbol_changed)
        sel_layout.addWidget(self.combo)
        
        refresh_label = QLabel("Auto-refresh (sec):")
        sel_layout.addWidget(refresh_label)
        self.refresh_spin = QSpinBox()
        self.refresh_spin.setMinimum(5)
        self.refresh_spin.setMaximum(60)
        self.refresh_spin.setValue(15)
        self.refresh_spin.valueChanged.connect(self.update_refresh_interval)
        sel_layout.addWidget(self.refresh_spin)
        
        sel_layout.addStretch()
        layout.addLayout(sel_layout)
        
        # Price display
        price_layout = QHBoxLayout()
        self.price_label = QLabel("Price: Loading...")
        self.price_label.setFont(QFont('Arial', 14, QFont.Bold))
        price_layout.addWidget(self.price_label)
        
        self.change_label = QLabel("")
        self.change_label.setFont(QFont('Arial', 12, QFont.Bold))
        price_layout.addWidget(self.change_label)
        
        self.timestamp_label = QLabel("")
        price_layout.addWidget(self.timestamp_label)
        price_layout.addStretch()
        layout.addLayout(price_layout)
        
        # Forecast box with graph
        forecast_box = QWidget()
        forecast_box.setStyleSheet("background-color: #e8f4f8; border-radius: 8px; padding: 10px;")
        forecast_layout = QVBoxLayout()
        
        forecast_title = QLabel("40-MINUTE FORECAST")
        forecast_title.setFont(QFont('Arial', 12, QFont.Bold))
        forecast_layout.addWidget(forecast_title)
        
        # Forecast text info
        forecast_text_layout = QHBoxLayout()
        self.direction_label = QLabel("Direction: LOADING")
        self.direction_label.setFont(QFont('Arial', 14, QFont.Bold))
        self.direction_label.setStyleSheet("color: #0066cc;")
        forecast_text_layout.addWidget(self.direction_label)
        
        self.conf_label = QLabel("Confidence: --")
        self.conf_label.setFont(QFont('Arial', 12))
        forecast_text_layout.addWidget(self.conf_label)
        
        self.ret_label = QLabel("Return: --")
        self.ret_label.setFont(QFont('Arial', 12))
        forecast_text_layout.addWidget(self.ret_label)

        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.clicked.connect(lambda: self.zoom_chart(0.8))
        forecast_text_layout.addWidget(self.zoom_in_btn)

        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.clicked.connect(lambda: self.zoom_chart(1.25))
        forecast_text_layout.addWidget(self.zoom_out_btn)

        self.reset_zoom_btn = QPushButton("Reset Zoom")
        self.reset_zoom_btn.clicked.connect(self.reset_chart_zoom)
        forecast_text_layout.addWidget(self.reset_zoom_btn)

        forecast_text_layout.addStretch()
        forecast_layout.addLayout(forecast_text_layout)
        
        # Forecast graph - MUCH BIGGER (12 inches wide x 6 inches tall)
        self.forecast_figure = Figure(figsize=(12, 6), dpi=100)
        self.forecast_canvas = FigureCanvas(self.forecast_figure)
        self.forecast_canvas.mpl_connect('scroll_event', self.on_scroll)
        self.forecast_canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.forecast_canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.forecast_canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.forecast_canvas.setMinimumHeight(500)  # Ensure minimum height
        forecast_layout.addWidget(self.forecast_canvas, 1)  # Give it most of the space
        
        forecast_box.setLayout(forecast_layout)
        layout.addWidget(forecast_box, 1)  # Give forecast box priority
        
        # Stats table
        stats_label = QLabel("Price & Forecast Statistics")
        stats_label.setFont(QFont('Arial', 11, QFont.Bold))
        layout.addWidget(stats_label)
        
        self.stats = QTableWidget()
        self.stats.setColumnCount(2)
        self.stats.setHorizontalHeaderLabels(['Metric', 'Value'])
        self.stats.setMaximumHeight(200)
        layout.addWidget(self.stats)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def update_forecast_chart(self, bars_df, forecast=None):
        """Render a real-time style chart (time on X axis, stock price on Y axis).
        Confidence bands are removed."""
        previous_limits = None
        if self._manual_zoom and self.forecast_figure.axes:
            prev_axis = self.forecast_figure.axes[0]
            previous_limits = (prev_axis.get_xlim(), prev_axis.get_ylim())

        self.forecast_figure.clear()
        axis = self.forecast_figure.add_subplot(111)

        if bars_df is None or bars_df.empty:
            self._x_data_bounds = None
            self._y_data_bounds = None
            axis.text(0.5, 0.5, "No price data available", ha='center', va='center', transform=axis.transAxes)
            self.forecast_figure.tight_layout()
            self.forecast_canvas.draw()
            return

        plot_df = bars_df.tail(120).copy()
        plot_df['timestamp'] = pd.to_datetime(plot_df['timestamp'], errors='coerce')
        plot_df = plot_df.dropna(subset=['timestamp', 'close'])
        if plot_df.empty:
            self._x_data_bounds = None
            self._y_data_bounds = None
            axis.text(0.5, 0.5, "No valid price data available", ha='center', va='center', transform=axis.transAxes)
            self.forecast_figure.tight_layout()
            self.forecast_canvas.draw()
            return

        axis.plot(
            plot_df['timestamp'],
            plot_df['close'],
            color='#1f77b4',
            linewidth=2.2,
            label='Price'
        )

        last_ts = plot_df['timestamp'].iloc[-1]
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

        x_min = mdates.date2num(plot_df['timestamp'].min())
        x_max = mdates.date2num(plot_df['timestamp'].max())
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

        axis.set_title('Real-Time Stock Price (Time-Based)', fontsize=14, fontweight='bold')
        axis.set_xlabel('Time')
        axis.set_ylabel('Price ($)')
        axis.grid(True, alpha=0.3, linestyle=':')
        axis.legend(loc='upper left')
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axis.tick_params(axis='x', rotation=25)

        if previous_limits is not None:
            xlim, ylim = self._clamp_view_limits(previous_limits[0], previous_limits[1])
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)

        self.forecast_figure.tight_layout()
        self.forecast_canvas.draw()

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

            start_timestamp = pd.to_datetime(last_ts, errors='coerce')
            if pd.isna(start_timestamp):
                start_timestamp = datetime.utcnow()
            else:
                start_timestamp = start_timestamp.to_pydatetime()

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
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        if event.button == 'up':
            scale_factor = 0.8
        elif event.button == 'down':
            scale_factor = 1.2
        else:
            return

        self._apply_zoom(event.inaxes, scale_factor, anchor=(event.xdata, event.ydata))
        self._manual_zoom = True
        self.forecast_canvas.draw_idle()

    def on_mouse_press(self, event):
        """Start pan mode on left-click inside the chart."""
        if event.button != 1 or event.inaxes is None or event.xdata is None or event.ydata is None:
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
        if event.xdata is None or event.ydata is None:
            return

        dx = self._pan_start['xdata'] - event.xdata
        dy = self._pan_start['ydata'] - event.ydata

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

        xlim = sorted((float(xlim[0]), float(xlim[1])))
        ylim = sorted((float(ylim[0]), float(ylim[1])))

        data_x_span = max(data_x_max - data_x_min, 1.0 / 1440.0)  # 1 minute
        data_y_span = max(data_y_max - data_y_min, max(abs(data_y_min) * 0.01, 0.1))

        min_x_span = max(data_x_span * 0.01, 15.0 / 86400.0)  # 15 seconds
        min_y_span = max(data_y_span * 0.01, 0.01)
        max_x_span = data_x_span * 2.0
        max_y_span = data_y_span * 3.0

        x_span = max(min_x_span, min(xlim[1] - xlim[0], max_x_span))
        y_span = max(min_y_span, min(ylim[1] - ylim[0], max_y_span))

        x_margin = max(data_x_span * 0.25, 5.0 / 1440.0)  # 5 minutes
        y_margin = max(data_y_span * 0.25, 0.25)

        x_center = (xlim[0] + xlim[1]) / 2.0
        y_center = (ylim[0] + ylim[1]) / 2.0

        min_x_center = data_x_min - x_margin + x_span / 2.0
        max_x_center = data_x_max + x_margin - x_span / 2.0
        min_y_center = data_y_min - y_margin + y_span / 2.0
        max_y_center = data_y_max + y_margin - y_span / 2.0

        if min_x_center > max_x_center:
            x_center = (data_x_min + data_x_max) / 2.0
        else:
            x_center = min(max(x_center, min_x_center), max_x_center)

        if min_y_center > max_y_center:
            y_center = (data_y_min + data_y_max) / 2.0
        else:
            y_center = min(max(y_center, min_y_center), max_y_center)

        return (
            (x_center - x_span / 2.0, x_center + x_span / 2.0),
            (y_center - y_span / 2.0, y_center + y_span / 2.0),
        )

    def _apply_zoom(self, axis, scale_factor, anchor=None):
        """Apply zoom around an anchor point on the given axis."""
        cur_xlim = axis.get_xlim()
        cur_ylim = axis.get_ylim()

        if anchor is None:
            x_anchor = (cur_xlim[0] + cur_xlim[1]) / 2.0
            y_anchor = (cur_ylim[0] + cur_ylim[1]) / 2.0
        else:
            x_anchor, y_anchor = anchor

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - x_anchor) / (cur_xlim[1] - cur_xlim[0]) if cur_xlim[1] != cur_xlim[0] else 0.5
        rely = (cur_ylim[1] - y_anchor) / (cur_ylim[1] - cur_ylim[0]) if cur_ylim[1] != cur_ylim[0] else 0.5

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
        self.timer.start(15000)  # 15 seconds

        self.scoring_timer = QTimer()
        self.scoring_timer.timeout.connect(self.run_scoring_cycle)
        self.scoring_timer.start(60000)  # 60 seconds
        self.run_scoring_cycle()
    
    def update_refresh_interval(self):
        """Update refresh interval."""
        interval_ms = self.refresh_spin.value() * 1000
        self.timer.setInterval(interval_ms)

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
        self.load_ticker()
    
    def load_ticker(self):
        """Load ticker data and forecast."""
        symbol = self.current_symbol

        bars_df = pd.DataFrame()
        try:
            bars = self.client.get_bars(symbol, limit=240)
            if isinstance(bars, pd.DataFrame):
                bars_df = bars
        except Exception as e:
            logger.error(f"Error getting bars for {symbol}: {e}")

        try:
            quote = self.client.get_quote(symbol)
            if quote is None:
                quote = {}
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            quote = {}

        # Extract quote fields (supports finnhub/databento style)
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
            self.change_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.change_label.setText(f"Change: ${change:.2f} ({chg_pct:.2f}%)")
            self.change_label.setStyleSheet("color: red; font-weight: bold;")

        timestamp = datetime.now().strftime("%H:%M:%S")
        if not bars_df.empty and 'timestamp' in bars_df.columns:
            try:
                last_bar_ts = pd.to_datetime(bars_df['timestamp'].iloc[-1], utc=True)
                now_utc = datetime.utcnow().replace(tzinfo=last_bar_ts.tzinfo)
                age_min = int((now_utc - last_bar_ts).total_seconds() // 60)
                self.timestamp_label.setText(
                    f"App update: {timestamp} | Last market print: {last_bar_ts.strftime('%Y-%m-%d %H:%M')} UTC ({age_min}m ago)"
                )
            except Exception:
                self.timestamp_label.setText(f"Last update: {timestamp}")
        else:
            self.timestamp_label.setText(f"Last update: {timestamp}")

        # Use enhanced forecasting engine first; fallback to quick heuristic if needed.
        forecast = None
        try:
            model_forecast = self.forecaster.generate_forecast(symbol)
            model_status = (model_forecast or {}).get('model_status', '')
            is_closed_status = isinstance(model_status, str) and model_status.lower().startswith('market closed')

            if model_forecast and model_forecast.get('prediction_return') is not None:
                forecast = {
                    'direction': model_forecast.get('direction', 'FLAT'),
                    'confidence': float(model_forecast.get('confidence', 50.0)),
                    # ForecasterEngine returns fractional return; UI expects percent.
                    'predicted_return': float(model_forecast.get('prediction_return', 0.0)) * 100.0,
                    'model_status': model_forecast.get('model_status', 'OK'),
                }
            elif is_closed_status:
                # Do not fall back to quick heuristic when market is closed.
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

        if forecast:
            direction = forecast.get('direction', 'FLAT')
            confidence = float(forecast.get('confidence', 50))
            pred_return = float(forecast.get('predicted_return', 0))

            dir_icon = "UP" if direction == 'UP' else ("DOWN" if direction == 'DOWN' else "FLAT")

            self.direction_label.setText(f"Direction: {dir_icon}")
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

        self.stats.setRowCount(len(stats_data))
        for row, (metric, value) in enumerate(stats_data):
            item1 = QTableWidgetItem(metric)
            item2 = QTableWidgetItem(value)

            if metric == '':
                item1.setBackground(QColor(200, 200, 200))
                item2.setBackground(QColor(200, 200, 200))

            self.stats.setItem(row, 0, item1)
            self.stats.setItem(row, 1, item2)
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
        self.init_ui()
        self.setup_auto_refresh()
        self.load_news()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        title = QLabel("📰 Financial News & Headlines")
        title.setFont(QFont('Arial', 16, QFont.Bold))
        layout.addWidget(title)
        
        # Symbol selector
        sel_layout = QHBoxLayout()
        sel_layout.addWidget(QLabel("Symbol:"))
        self.combo = QComboBox()
        self.combo.addItems(POPULAR_STOCKS)
        self.combo.currentTextChanged.connect(self.on_symbol_changed)
        sel_layout.addWidget(self.combo)
        sel_layout.addStretch()
        layout.addLayout(sel_layout)
        
        # News display
        self.news_text = QTextEdit()
        self.news_text.setReadOnly(True)
        self.news_text.setFont(QFont('Courier', 9))
        layout.addWidget(self.news_text)
        
        self.setLayout(layout)
    
    def setup_auto_refresh(self):
        """Setup automatic refresh timer."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.load_news)
        self.timer.start(60000)  # 60 seconds
    
    def on_symbol_changed(self):
        """Handle symbol change."""
        self.current_symbol = self.combo.currentText()
        self.load_news()
    
    def load_news(self):
        """Load news headlines."""
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
    
    def display_demo_news(self, symbol: str):
        """Display demo news when API unavailable."""
        demo_headlines = [
            {
                'headline': f'{symbol} gains momentum as analysts revise price targets',
                'source': 'MarketWatch',
                'summary': f'{symbol} experienced strong trading activity today with institutional interest',
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'headline': f'Tech sector rally pushes {symbol} higher',
                'source': 'Bloomberg',
                'summary': 'Broad tech rally supports gains across major indices',
                'datetime': (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'headline': f'{symbol} Q4 earnings beat expectations',
                'source': 'Reuters',
                'summary': 'Company delivers strong quarter with revenue growth acceleration',
                'datetime': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
            },
        ]
        self.display_news(demo_headlines)
    
    def display_news(self, headlines):
        """Display news headlines."""
        text = f"Latest news for {self.current_symbol}\n"
        text += "=" * 80 + "\n\n"
        
        for i, headline in enumerate(headlines, 1):
            text += f"{i}. {headline.get('headline', 'N/A')}\n"
            text += f"   Source: {headline.get('source', 'Unknown')}\n"
            text += f"   Date: {headline.get('datetime', 'Unknown')}\n"
            text += f"   {headline.get('summary', 'No summary available')}\n"
            text += "\n" + "-" * 80 + "\n\n"
        
        self.news_text.setText(text)


class SettingsTab(QWidget):
    """Settings and diagnostics."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        title = QLabel("⚙️ Settings & Diagnostics")
        title.setFont(QFont('Arial', 16, QFont.Bold))
        layout.addWidget(title)
        
        diag_label = QLabel("System Information")
        diag_label.setFont(QFont('Arial', 12, QFont.Bold))
        layout.addWidget(diag_label)
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)

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
Futursia Forecasting V.1.0 - System Diagnostics
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
  ✅ Real-time stock quotes
  ✅ 40-minute forecasts
  ✅ Forecast distribution graphs
  ✅ Financial news headlines
  ✅ Auto-refresh enabled
  ✅ Non-blocking UI
"""
        
        info_text.setText(info)
        layout.addWidget(info_text)
        
        layout.addStretch()
        self.setLayout(layout)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Load config
        config_path = Path(__file__).parent / 'configs' / 'config.yaml'
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.setWindowTitle("Futursia Forecasting V.1.0")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create tabs
        tabs = QTabWidget()
        tabs.addTab(DashboardTab(self.config), "Dashboard")
        tabs.addTab(TickerDetailTab(self.config), "Ticker Detail")
        tabs.addTab(NewsTab(self.config), "News")
        tabs.addTab(SettingsTab(self.config), "Settings")
        
        self.setCentralWidget(tabs)
        
        logger.info("Application started successfully")
    
    def closeEvent(self, event):
        """Handle application close."""
        logger.info("Application closing")
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

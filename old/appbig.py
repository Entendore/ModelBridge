#!/usr/bin/env python3
"""
LocalAI Studio Pro – Complete Local LLM Manager, Inference, & API Server Application
Includes: Mode Management, Smart GPU Offloading, Model Catalog with Stats
"""

import sys, os, json, time, re, shutil, subprocess, platform, hashlib, traceback, webbrowser
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from typing import List, Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QLabel, QPushButton, QLineEdit, QTextEdit,
    QComboBox, QProgressBar, QSlider, QSpinBox, QCheckBox, QDoubleSpinBox,
    QGroupBox, QFileDialog, QMessageBox, QSplitter, QFrame,
    QScrollArea, QSizePolicy, QTabWidget,
    QHeaderView, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QDialog, QDialogButtonBox, QFormLayout, QMenu,
    QListWidget, QListWidgetItem, QPlainTextEdit, QInputDialog,
    QSystemTrayIcon, QTextBrowser, QToolButton, QColorDialog,
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QSize, QTimer, QUrl, QRect, QProcess, QPoint,
)
from PySide6.QtGui import (
    QIcon, QFont, QColor, QPixmap, QPainter, QPen, QBrush,
    QLinearGradient, QAction, QCursor,
)

import requests
import psutil

# ── Optional inference backends ──────────────────────────────────────
try:
    from llama_cpp import Llama as LlamaCPP
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

# ── Optional FastAPI for API Server ──────────────────────────────────
try:
    import uvicorn
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel as PydanticModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# ── Markdown Rendering ───────────────────────────────────────────────
try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False


# ═══════════════════════════════════════════════════════════════════════
#  CONSTANTS & THEME SYSTEM
# ═══════════════════════════════════════════════════════════════════════
APP_NAME = "LocalAI Studio Pro"
APP_VERSION = "3.2.0"
ORG_NAME = "LocalAI"

DEFAULT_MODELS_DIR = str(Path.home() / ".localai" / "models")
CONFIG_FILE = str(Path.home() / ".localai" / "config.json")
PRESETS_DIR = str(Path.home() / ".localai" / "sampling_presets")
COLOR_PRESETS_DIR = str(Path.home() / ".localai" / "color_presets")
MODES_DIR = str(Path.home() / ".localai" / "modes")

HF_API = "https://huggingface.co/api"
MS_API = "https://modelscope.cn/api/v1"
GH_API = "https://api.github.com"

DARK_THEME = {
    "bg":              "#0d1117",
    "bg_alt":          "#010409",
    "surface":         "#161b22",
    "surface_hover":   "#1f2937",
    "border":          "#21262d",
    "border_focus":    "#30363d",
    "text":            "#e6edf3",
    "text_muted":      "#8b949e",
    "accent":          "#58a6ff",
    "primary":         "#238636",
    "primary_hover":   "#2ea043",
    "danger":          "#da3633",
    "danger_hover":    "#f85149",
    "warning":         "#9e6a03",
    "warning_hover":   "#bb8009",
    "secondary_bg":    "#21262d",
    "secondary_hover": "#30363d",
    "user_bubble":     "#1c2333",
    "scrollbar":       "#30363d",
}

LIGHT_THEME = {
    "bg":              "#ffffff",
    "bg_alt":          "#f6f8fa",
    "surface":         "#f6f8fa",
    "surface_hover":   "#eaeef2",
    "border":          "#d0d7de",
    "border_focus":    "#afb8c1",
    "text":            "#1f2328",
    "text_muted":      "#656d76",
    "accent":          "#0969da",
    "primary":         "#1a7f37",
    "primary_hover":   "#1a7f37",
    "danger":          "#cf222e",
    "danger_hover":    "#a40e26",
    "warning":         "#9a6700",
    "warning_hover":   "#7c5500",
    "secondary_bg":    "#f6f8fa",
    "secondary_hover": "#eaeef2",
    "user_bubble":     "#ddf4ff",
    "scrollbar":       "#afb8c1",
}


class ThemeManager:
    PRESETS_DIR = COLOR_PRESETS_DIR

    def __init__(self):
        self._themes = OrderedDict([
            ("Dark",  dict(DARK_THEME)),
            ("Light", dict(LIGHT_THEME)),
        ])
        self._custom_themes: OrderedDict = OrderedDict()
        self._load_custom_themes()

    @property
    def theme_names(self) -> list:
        return list(self._themes.keys()) + list(self._custom_themes.keys())

    def get_theme(self, name: str) -> dict:
        if name in self._themes: return dict(self._themes[name])
        if name in self._custom_themes: return dict(self._custom_themes[name])
        return dict(DARK_THEME)

    def _load_custom_themes(self):
        d = Path(self.PRESETS_DIR); d.mkdir(parents=True, exist_ok=True)
        for f in sorted(d.glob("*.json")):
            try:
                with open(f) as fh: data = json.load(fh)
                name = data.get("name", f.stem); colors = data.get("colors", {})
                if set(colors.keys()) == set(DARK_THEME.keys()): self._custom_themes[name] = colors
            except Exception: pass

    def save_preset(self, name: str, colors: dict) -> str:
        d = Path(self.PRESETS_DIR); d.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r'[^\w\-.]', '_', name); path = d / f"{safe}.json"
        with open(path, "w") as f: json.dump({"name": name, "colors": colors}, f, indent=2)
        self._custom_themes[name] = dict(colors)
        return str(path)

    def delete_preset(self, name: str) -> bool:
        if name not in self._custom_themes: return False
        safe = re.sub(r'[^\w\-.]', '_', name); path = Path(self.PRESETS_DIR) / f"{safe}.json"
        if path.exists(): path.unlink()
        del self._custom_themes[name]; return True

    def export_preset(self, name: str, dest: str) -> bool:
        colors = self.get_theme(name)
        try:
            with open(dest, "w") as f: json.dump({"name": name, "colors": colors}, f, indent=2)
            return True
        except Exception: return False

    def import_preset(self, filepath: str) -> str:
        try:
            with open(filepath) as f: data = json.load(f)
            name = data.get("name", Path(filepath).stem); colors = data.get("colors", {})
            if set(colors.keys()) != set(DARK_THEME.keys()): raise ValueError("Invalid color keys")
            self.save_preset(name, colors); return name
        except Exception as e: raise ValueError(f"Import failed: {e}")


class PresetManager:
    PRESET_KEYS = [
        "temperature", "top_p", "min_p", "max_tokens",
        "frequency_penalty", "presence_penalty",
        "mirostat_mode", "mirostat_tau", "mirostat_eta",
    ]
    DEFAULT_PRESETS = {
        "Creative": {"temperature": 0.9, "top_p": 0.95, "min_p": 0.05, "max_tokens": 4096, "frequency_penalty": 0.0, "presence_penalty": 0.0, "mirostat_mode": 0, "mirostat_tau": 5.0, "mirostat_eta": 0.1},
        "Precise": {"temperature": 0.3, "top_p": 0.85, "min_p": 0.1, "max_tokens": 2048, "frequency_penalty": 0.0, "presence_penalty": 0.0, "mirostat_mode": 0, "mirostat_tau": 5.0, "mirostat_eta": 0.1},
        "Balanced": {"temperature": 0.7, "top_p": 0.9, "min_p": 0.05, "max_tokens": 2048, "frequency_penalty": 0.0, "presence_penalty": 0.0, "mirostat_mode": 0, "mirostat_tau": 5.0, "mirostat_eta": 0.1},
        "Mirostat": {"temperature": 1.0, "top_p": 1.0, "min_p": 1.0, "max_tokens": 2048, "frequency_penalty": 0.0, "presence_penalty": 0.0, "mirostat_mode": 2, "mirostat_tau": 5.0, "mirostat_eta": 0.1},
    }

    def __init__(self):
        Path(PRESETS_DIR).mkdir(parents=True, exist_ok=True); self._ensure_defaults()

    def _ensure_defaults(self):
        for name, vals in self.DEFAULT_PRESETS.items():  # <-- FIXED: DEFAULT_PRESETS instead of DEFAULT_MODES
            p = Path(PRESETS_DIR) / f"{name}.json"
            if not p.exists():
                with open(p, "w") as f: json.dump({"name": name, "params": vals}, f, indent=2)

    def list_presets(self) -> list:
        presets = []
        for f in sorted(Path(PRESETS_DIR).glob("*.json")):
            try:
                with open(f) as fh: data = json.load(fh)
                presets.append((data.get("name", f.stem), data.get("params", {}), str(f)))
            except Exception: pass
        return presets

    def save_preset(self, name: str, params: dict) -> str:
        safe = re.sub(r'[^\w\-.]', '_', name); path = Path(PRESETS_DIR) / f"{safe}.json"
        with open(path, "w") as f: json.dump({"name": name, "params": {k: params.get(k) for k in self.PRESET_KEYS}}, f, indent=2)
        return str(path)

    def delete_preset(self, name: str) -> bool:
        safe = re.sub(r'[^\w\-.]', '_', name); path = Path(PRESETS_DIR) / f"{safe}.json"
        if path.exists(): path.unlink(); return True
        return False

    def load_preset(self, name: str) -> dict:
        for pname, params, _ in self.list_presets():
            if pname == name: return dict(params)
        return {}

    def export_preset(self, name: str, dest: str) -> bool:
        params = self.load_preset(name)
        if not params: return False
        try:
            with open(dest, "w") as f: json.dump({"name": name, "params": params}, f, indent=2)
            return True
        except Exception: return False

    def import_preset(self, filepath: str) -> str:
        with open(filepath) as f: data = json.load(f)
        name = data.get("name", Path(filepath).stem); params = data.get("params", {})
        self.save_preset(name, params); return name


def generate_stylesheet(c: dict) -> str:
    return f"""
QMainWindow {{ background-color: {c['bg']}; }}
QWidget {{ color: {c['text']}; font-family: 'Segoe UI', sans-serif; font-size: 13px; }}
QFrame#sidebar {{ background-color: {c['bg_alt']}; border-right: 1px solid {c['border']}; }}
QPushButton#navBtn {{
    background: transparent; border: none; border-radius: 8px;
    padding: 10px 16px; text-align: left; color: {c['text_muted']}; font-size: 13px;
}}
QPushButton#navBtn:hover {{ background: {c['surface']}; color: {c['text']}; }}
QPushButton#navBtn[active="true"] {{ background: {c['surface_hover']}; color: {c['accent']}; font-weight: 600; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {c['bg']}; border: 1px solid {c['border_focus']}; border-radius: 6px;
    padding: 8px 12px; color: {c['text']};
}}
QLineEdit:focus, QComboBox:focus {{ border-color: {c['accent']}; }}
QTextEdit, QPlainTextEdit, QTextBrowser {{
    background: {c['bg']}; border: 1px solid {c['border_focus']}; border-radius: 6px;
    padding: 8px; color: {c['text']};
}}
QPushButton#primaryBtn {{ background: {c['primary']}; border: none; border-radius: 6px; padding: 8px 20px; color: #fff; font-weight: 600; }}
QPushButton#primaryBtn:hover {{ background: {c['primary_hover']}; }}
QPushButton#dangerBtn {{ background: {c['danger']}; border: none; border-radius: 6px; padding: 8px 20px; color: #fff; font-weight: 600; }}
QPushButton#dangerBtn:hover {{ background: {c['danger_hover']}; }}
QPushButton#secondaryBtn {{ background: {c['secondary_bg']}; border: 1px solid {c['border_focus']}; border-radius: 6px; padding: 8px 20px; color: {c['text']}; }}
QPushButton#secondaryBtn:hover {{ background: {c['secondary_hover']}; }}
QPushButton#warningBtn {{ background: {c['warning']}; border: none; border-radius: 6px; padding: 8px 20px; color: #fff; font-weight: 600; }}
QPushButton#warningBtn:hover {{ background: {c['warning_hover']}; }}
QGroupBox {{ border: 1px solid {c['border']}; border-radius: 8px; margin-top: 12px; padding-top: 16px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {c['text']}; }}
QProgressBar {{ background: {c['secondary_bg']}; border: none; border-radius: 4px; height: 8px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {c['primary']}; border-radius: 4px; }}
QSlider::groove:horizontal {{ background: {c['secondary_bg']}; height: 6px; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {c['accent']}; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; }}
QTabWidget::pane {{ border: 1px solid {c['border']}; border-radius: 6px; }}
QTabBar::tab {{ background: {c['bg']}; border: 1px solid {c['border']}; padding: 8px 20px; border-top-left-radius: 6px; border-top-right-radius: 6px; color: {c['text_muted']}; }}
QTabBar::tab:selected {{ background: {c['surface']}; color: {c['accent']}; border-bottom: 2px solid {c['accent']}; }}
QScrollArea {{ border: none; background: transparent; }}
QTableWidget {{ background: {c['bg']}; gridline-color: {c['border']}; border: 1px solid {c['border']}; border-radius: 6px; }}
QHeaderView::section {{ background: {c['surface']}; border: 1px solid {c['border']}; padding: 6px; font-weight: 600; color: {c['text_muted']}; }}
QScrollBar:vertical {{ background: {c['bg']}; width: 10px; border: none; }}
QScrollBar::handle:vertical {{ background: {c['scrollbar']}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QListWidget {{ background: {c['bg']}; border: 1px solid {c['border']}; border-radius: 6px; }}
QListWidget::item {{ padding: 6px; border-bottom: 1px solid {c['surface']}; }}
QListWidget::item:selected {{ background: {c['surface_hover']}; color: {c['accent']}; }}
QCheckBox {{ spacing: 8px; color: {c['text']}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {c['border_focus']}; border-radius: 3px; background: {c['bg']}; }}
QCheckBox::indicator:checked {{ background: {c['accent']}; border-color: {c['accent']}; }}
QMenu {{ background: {c['surface']}; border: 1px solid {c['border_focus']}; color: {c['text']}; }}
QMenu::item:selected {{ background: {c['surface_hover']}; }}
QDialog {{ background: {c['bg']}; }}
"""


# ═══════════════════════════════════════════════════════════════════════
#  CONFIG & UTILS
# ═══════════════════════════════════════════════════════════════════════
class AppConfig:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        Path(CONFIG_FILE).parent.mkdir(parents=True, exist_ok=True)
        if Path(CONFIG_FILE).exists():
            try: 
                with open(CONFIG_FILE) as f: self._data = json.load(f)
            except Exception: self._data = {}
        else: self._data = {}

    def save(self):
        Path(CONFIG_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f: json.dump(self._data, f, indent=2)

    def get(self, key, default=None): return self._data.get(key, default)
    def set(self, key, value): self._data[key] = value; self.save()

    @property
    def models_dir(self):
        d = self.get("models_dir", DEFAULT_MODELS_DIR)
        Path(d).mkdir(parents=True, exist_ok=True); return d
    @models_dir.setter
    def models_dir(self, v): self.set("models_dir", v)

    @property
    def outputs_dir(self) -> str:
        d = self.get("outputs_dir", str(Path.home() / ".localai" / "outputs"))
        Path(d).mkdir(parents=True, exist_ok=True); return d
    @outputs_dir.setter
    def outputs_dir(self, v: str): self.set("outputs_dir", v)

    @property
    def theme_name(self) -> str: return self.get("theme_name", "Dark")
    @theme_name.setter
    def theme_name(self, v: str): self.set("theme_name", v)

    @property
    def inference_backend(self): return self.get("inference_backend", "llama_cpp")
    @property
    def ollama_url(self): return self.get("ollama_url", "http://localhost:11434")
    @property
    def llama_server_url(self): return self.get("llama_server_url", "http://localhost:8080")
    @property
    def n_ctx(self): return self.get("n_ctx", 4096)
    @property
    def n_gpu_layers(self): return self.get("n_gpu_layers", 0)
    @property
    def n_threads(self): return self.get("n_threads", max(1, (psutil.cpu_count(logical=False) or 2) - 1))
    @property
    def min_p(self): return self.get("min_p", 0.05)
    @property
    def frequency_penalty(self): return self.get("frequency_penalty", 0.0)
    @property
    def presence_penalty(self): return self.get("presence_penalty", 0.0)
    @property
    def mirostat_mode(self): return self.get("mirostat_mode", 0)
    @property
    def mirostat_tau(self): return self.get("mirostat_tau", 5.0)
    @property
    def mirostat_eta(self): return self.get("mirostat_eta", 0.1)
    @property
    def temp(self): return self.get("temperature", 0.7)
    @property
    def top_p(self): return self.get("top_p", 0.9)
    @property
    def max_tokens(self): return self.get("max_tokens", 2048)
    @property
    def server_port(self): return self.get("server_port", 8081)
    @property
    def api_key(self): return self.get("api_key", "sk-localai")
    @property
    def cors_enabled(self): return self.get("cors_enabled", True)
    @property
    def auto_gpu_layers(self): return self.get("auto_gpu_layers", True)
    @property
    def vram_headroom_pct(self): return self.get("vram_headroom_pct", 15)
    @property
    def warn_vram_exceed(self): return self.get("warn_vram_exceed", True)
    @property
    def n_batch(self): return self.get("n_batch", 512)

    def apply_preset(self, params: dict):
        for key in PresetManager.PRESET_KEYS:
            if key in params: self.set(key, params[key])

    def extract_preset(self) -> dict:
        return {key: self.get(key, PresetManager.DEFAULT_PRESETS["Balanced"].get(key)) for key in PresetManager.PRESET_KEYS}


def fmt_size(n: int) -> str:
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024: return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"

def fmt_speed(kbps: float) -> str:
    if kbps < 1024: return f"{kbps:.0f} KB/s"
    return f"{kbps/1024:.1f} MB/s"


# ═══════════════════════════════════════════════════════════════════════
#  MODE MANAGER
# ═══════════════════════════════════════════════════════════════════════
class ModeManager:
    MODE_KEYS = [
        "mode_name", "model_id", "system_prompt", "n_ctx",
        "n_gpu_layers", "n_threads", "temperature", "top_p", "min_p",
        "max_tokens", "frequency_penalty", "presence_penalty",
        "mirostat_mode", "mirostat_tau", "mirostat_eta",
        "context_template", "stop_sequences", "grammar_file",
    ]

    DEFAULT_MODES = {
        "Chat": {
            "mode_name": "Chat", "model_id": "", "system_prompt": "You are a helpful, friendly AI assistant.",
            "n_ctx": 4096, "n_gpu_layers": -1, "n_threads": 4, "temperature": 0.7, "top_p": 0.9,
            "min_p": 0.05, "max_tokens": 2048, "frequency_penalty": 0.0, "presence_penalty": 0.0,
            "mirostat_mode": 0, "mirostat_tau": 5.0, "mirostat_eta": 0.1, "context_template": "chat",
            "stop_sequences": [], "grammar_file": "",
        },
        "Code": {
            "mode_name": "Code", "model_id": "", "system_prompt": "You are an expert programmer. Respond with clean, well-commented code. Use markdown code blocks.",
            "n_ctx": 8192, "n_gpu_layers": -1, "n_threads": 4, "temperature": 0.2, "top_p": 0.85,
            "min_p": 0.1, "max_tokens": 4096, "frequency_penalty": 0.0, "presence_penalty": 0.0,
            "mirostat_mode": 0, "mirostat_tau": 5.0, "mirostat_eta": 0.1, "context_template": "code",
            "stop_sequences": ["```"], "grammar_file": "",
        },
        "Creative Writing": {
            "mode_name": "Creative Writing", "model_id": "", "system_prompt": "You are a creative writer with a vivid imagination. Write with rich detail and engaging prose.",
            "n_ctx": 8192, "n_gpu_layers": -1, "n_threads": 4, "temperature": 1.0, "top_p": 0.95,
            "min_p": 0.02, "max_tokens": 4096, "frequency_penalty": 0.3, "presence_penalty": 0.2,
            "mirostat_mode": 2, "mirostat_tau": 5.0, "mirostat_eta": 0.1, "context_template": "chat",
            "stop_sequences": [], "grammar_file": "",
        },
        "RAG / QA": {
            "mode_name": "RAG / QA", "model_id": "", "system_prompt": "Answer questions using only the provided context. If the answer is not in the context, say so honestly.",
            "n_ctx": 16384, "n_gpu_layers": -1, "n_threads": 4, "temperature": 0.1, "top_p": 0.8,
            "min_p": 0.15, "max_tokens": 2048, "frequency_penalty": 0.0, "presence_penalty": 0.0,
            "mirostat_mode": 0, "mirostat_tau": 5.0, "mirostat_eta": 0.1, "context_template": "qa",
            "stop_sequences": [], "grammar_file": "",
        },
        "Summarization": {
            "mode_name": "Summarization", "model_id": "", "system_prompt": "Summarize the provided text concisely. Focus on key points and main ideas.",
            "n_ctx": 16384, "n_gpu_layers": -1, "n_threads": 4, "temperature": 0.3, "top_p": 0.85,
            "min_p": 0.1, "max_tokens": 1024, "frequency_penalty": 0.0, "presence_penalty": 0.0,
            "mirostat_mode": 0, "mirostat_tau": 5.0, "mirostat_eta": 0.1, "context_template": "summarize",
            "stop_sequences": [], "grammar_file": "",
        },
    }

    @staticmethod
    def _safe_filename(name: str) -> str:
        """Sanitize a mode name into a safe filename (no slashes, spaces, etc.)."""
        return re.sub(r'[^\w\-.]', '_', name)

    def __init__(self):
        Path(MODES_DIR).mkdir(parents=True, exist_ok=True)
        self._ensure_defaults()

    def _ensure_defaults(self):
        for name, vals in self.DEFAULT_MODES.items():
            safe = self._safe_filename(name)
            p = Path(MODES_DIR) / f"{safe}.json"
            if not p.exists():
                with open(p, "w") as f:
                    json.dump(vals, f, indent=2)

    def list_modes(self) -> list:
        modes = []
        for f in sorted(Path(MODES_DIR).glob("*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                modes.append((data.get("mode_name", f.stem), data, str(f)))
            except Exception:
                pass
        return modes

    def get_mode(self, name: str) -> dict:
        for mname, data, _ in self.list_modes():
            if mname == name:
                return dict(data)
        return dict(self.DEFAULT_MODES["Chat"])

    def save_mode(self, mode_data: dict) -> str:
        name = mode_data.get("mode_name", "Custom")
        safe = self._safe_filename(name)
        path = Path(MODES_DIR) / f"{safe}.json"
        with open(path, "w") as f:
            json.dump(mode_data, f, indent=2)
        return str(path)

    def delete_mode(self, name: str) -> bool:
        safe = self._safe_filename(name)
        path = Path(MODES_DIR) / f"{safe}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def apply_mode_to_config(self, name: str, config: AppConfig):
        data = self.get_mode(name)
        skip = {"mode_name", "model_id", "context_template", "stop_sequences", "grammar_file"}
        for key, value in data.items():
            if key not in skip:
                config.set(key, value)
        return data.get("model_id", ""), data.get("system_prompt", ""), data.get("context_template", "chat")


# ═══════════════════════════════════════════════════════════════════════
#  SMART GPU MANAGER
# ═══════════════════════════════════════════════════════════════════════
class GPUDetector:
    @staticmethod
    def detect_gpus() -> list:
        gpus = []
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                for line in r.stdout.strip().split("\n"):
                    p = [x.strip() for x in line.split(",")]
                    if len(p) >= 5:
                        gpus.append({"type": "nvidia", "index": int(p[0]), "name": p[1], "total_vram_mb": float(p[2]), "free_vram_mb": float(p[3]), "compute_cap": p[4], "driver": p[5] if len(p) > 5 else ""})
        except Exception: pass
        
        if not gpus and platform.system() == "Darwin":
            try:
                r = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=10)
                if r.returncode == 0 and "Metal" in r.stdout:
                    mem = psutil.virtual_memory()
                    gpus.append({"type": "apple_metal", "index": 0, "name": "Apple Metal (Unified Memory)", "total_vram_mb": mem.total / 1e6, "free_vram_mb": mem.available / 1e6, "compute_cap": "metal", "driver": ""})
            except Exception: pass
        return gpus


class VRAMEstimator:
    BYTES_PER_PARAM = {"q4_0": 0.56, "q4_1": 0.60, "q5_0": 0.65, "q5_1": 0.70, "q8_0": 1.06, "q2_k": 0.38, "q3_k_s": 0.42, "q3_k_m": 0.48, "q3_k_l": 0.52, "q4_k_s": 0.56, "q4_k_m": 0.60, "q5_k_s": 0.65, "q5_k_m": 0.68, "q6_k": 0.78, "f16": 2.0, "f32": 4.0}

    @staticmethod
    def estimate_from_file_size(file_size_bytes: int, n_ctx: int = 4096) -> dict:
        model_vram = file_size_bytes
        kv_cache_mb = (n_ctx * 2 * 32 * 4096) / (1024 * 1024)
        overhead_mb = 100
        total_vram = model_vram + (kv_cache_mb * 1024 * 1024) + (overhead_mb * 1024 * 1024)
        return {"model_size_bytes": file_size_bytes, "model_vram_bytes": model_vram, "kv_cache_bytes": int(kv_cache_mb * 1024 * 1024), "overhead_bytes": int(overhead_mb * 1024 * 1024), "total_vram_bytes": int(total_vram), "total_vram_gb": total_vram / (1024**3)}

    @staticmethod
    def estimate_from_params(n_params_billion: float, quant_type: str = "q4_k_m", n_ctx: int = 4096) -> dict:
        bpp = VRAMEstimator.BYTES_PER_PARAM.get(quant_type, 0.60)
        model_vram = n_params_billion * 1e9 * bpp
        kv_cache_mb = n_params_billion * n_ctx * 0.002
        total = model_vram + (kv_cache_mb * 1024 * 1024) + (100 * 1024 * 1024)
        return {"n_params_billion": n_params_billion, "quant_type": quant_type, "model_vram_gb": model_vram / (1024**3), "kv_cache_gb": kv_cache_mb / 1024, "total_vram_gb": total / (1024**3)}

    @staticmethod
    def calculate_max_layers(model_size_bytes: int, n_layers: int, available_vram_bytes: int, n_ctx: int = 4096) -> int:
        layer_size = model_size_bytes / n_layers if n_layers > 0 else 0
        kv_bytes = int(n_ctx * 2 * 32 * 4096)
        usable = available_vram_bytes - kv_bytes - (100 * 1024 * 1024)
        if usable <= 0: return 0
        return min(int(usable / layer_size), n_layers)


class SmartGPUManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.gpus = GPUDetector.detect_gpus()

    def refresh(self): self.gpus = GPUDetector.detect_gpus()

    def get_optimal_gpu_layers(self, model_path: str, n_ctx: int = None) -> dict:
        if n_ctx is None: n_ctx = self.config.n_ctx
        model_path = Path(model_path)
        if not model_path.exists(): return {"n_gpu_layers": 0, "strategy": "cpu_only", "reason": "Model file not found"}
        
        model_size = model_path.stat().st_size
        if not self.gpus: return {"n_gpu_layers": 0, "strategy": "cpu_only", "reason": "No GPU detected"}
        
        best_gpu = max(self.gpus, key=lambda g: g.get("free_vram_mb", 0))
        free_vram = best_gpu.get("free_vram_mb", 0) * 1024 * 1024
        estimate = VRAMEstimator.estimate_from_file_size(model_size, n_ctx)
        total_needed = estimate["total_vram_bytes"]
        headroom = 1.0 - (self.config.vram_headroom_pct / 100.0)

        result = {"model_size": model_size, "model_size_gb": model_size / (1024**3), "gpu_name": best_gpu.get("name", "Unknown"), "gpu_free_vram_gb": free_vram / (1024**3), "estimated_vram_needed_gb": estimate["total_vram_gb"], "gpu_type": best_gpu.get("type", "unknown"), "gpu_index": best_gpu.get("index", 0)}

        if total_needed <= free_vram * headroom:
            result.update({"n_gpu_layers": -1, "strategy": "full_offload", "reason": f"Model fits in VRAM ({estimate['total_vram_gb']:.1f}GB < {free_vram/(1024**3):.1f}GB free)"})
        elif free_vram > 500 * 1024 * 1024:
            estimated_layers = self._estimate_n_layers(model_size)
            max_layers = VRAMEstimator.calculate_max_layers(model_size, estimated_layers, int(free_vram * headroom), n_ctx)
            result.update({"n_gpu_layers": max(0, max_layers), "strategy": "partial_offload", "reason": f"Partial offload: {max_layers}/{estimated_layers} layers to GPU", "estimated_total_layers": estimated_layers})
        else:
            result.update({"n_gpu_layers": 0, "strategy": "cpu_only", "reason": "Insufficient VRAM for any offloading"})
        return result

    def _estimate_n_layers(self, model_size_bytes: int) -> int:
        size_gb = model_size_bytes / (1024**3)
        if size_gb < 2: return 24
        elif size_gb < 6: return 32
        elif size_gb < 10: return 40
        elif size_gb < 22: return 60
        else: return 80

    def get_vram_warning(self, model_path: str, n_ctx: int = None, n_gpu_layers: int = -1) -> Optional[str]:
        if not self.gpus: return None
        model_path = Path(model_path)
        if not model_path.exists(): return None
        if n_ctx is None: n_ctx = self.config.n_ctx
        model_size = model_path.stat().st_size
        estimate = VRAMEstimator.estimate_from_file_size(model_size, n_ctx)
        best_gpu = max(self.gpus, key=lambda g: g.get("free_vram_mb", 0))
        free_vram = best_gpu.get("free_vram_mb", 0) * 1024 * 1024
        if n_gpu_layers == -1 and estimate["total_vram_bytes"] > free_vram * 0.90:
            return f"⚠️ This model requires ~{estimate['total_vram_gb']:.1f} GB VRAM but only {free_vram/(1024**3):.1f} GB is free on {best_gpu['name']}. Consider partial GPU offload or reducing context size."
        return None


# ═══════════════════════════════════════════════════════════════════════
#  MODEL CATALOG & STATS
# ═══════════════════════════════════════════════════════════════════════
RECOMMENDED_MODELS = [
    {"id": "llama-3.1-8b-instruct", "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "family": "Llama 3.1", "n_params_billion": 8.0, "context_length": 131072, "description": "Meta's Llama 3.1 8B Instruct — excellent general-purpose model.", "tags": ["chat", "instruct", "reasoning", "coding", "popular"], "license": "Llama 3.1 Community License", "author": "Meta", "variants": [{"name": "Q4_K_M", "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "size_gb": 4.92, "quant": "q4_k_m", "quality": "★★★★☆", "recommended": True}, {"name": "Q5_K_M", "filename": "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf", "size_gb": 5.68, "quant": "q5_k_m", "quality": "★★★★★", "recommended": False}], "benchmark": {"mmlu": 68.4, "humaneval": 62.2, "gsm8k": 76.6, "avg_vram_q4km_gb": 5.8, "speed_tok_s": 45}, "min_vram_gb": 6.0, "recommended_for": ["Chat", "RAG / QA", "Summarization"]},
    {"id": "mistral-7b-instruct-v0.3", "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF", "family": "Mistral", "n_params_billion": 7.25, "context_length": 32768, "description": "Mistral 7B Instruct v0.3 — efficient and fast, great for general chat.", "tags": ["chat", "instruct", "coding", "fast", "popular"], "license": "Apache 2.0", "author": "Mistral AI", "variants": [{"name": "Q4_K_M", "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", "size_gb": 4.37, "quant": "q4_k_m", "quality": "★★★★☆", "recommended": True}], "benchmark": {"mmlu": 63.6, "humaneval": 50.3, "gsm8k": 55.5, "avg_vram_q4km_gb": 5.2, "speed_tok_s": 55}, "min_vram_gb": 5.5, "recommended_for": ["Chat", "Code"]},
    {"id": "qwen2.5-7b-instruct", "repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF", "family": "Qwen 2.5", "n_params_billion": 7.6, "context_length": 131072, "description": "Qwen 2.5 7B Instruct — strong multilingual and coding performance.", "tags": ["chat", "instruct", "coding", "multilingual", "popular"], "license": "Apache 2.0", "author": "Alibaba", "variants": [{"name": "Q4_K_M", "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "size_gb": 4.68, "quant": "q4_k_m", "quality": "★★★★☆", "recommended": True}], "benchmark": {"mmlu": 74.3, "humaneval": 72.8, "gsm8k": 84.2, "avg_vram_q4km_gb": 5.5, "speed_tok_s": 48}, "min_vram_gb": 6.0, "recommended_for": ["Chat", "Code", "RAG / QA"]},
    {"id": "phi-3.5-mini-instruct", "repo_id": "bartowski/Phi-3.5-mini-instruct-GGUF", "family": "Phi 3.5", "n_params_billion": 3.8, "context_length": 131072, "description": "Microsoft Phi-3.5 Mini — very small and fast, perfect for CPU.", "tags": ["chat", "instruct", "small", "fast", "cpu-friendly"], "license": "MIT", "author": "Microsoft", "variants": [{"name": "Q4_K_M", "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf", "size_gb": 2.40, "quant": "q4_k_m", "quality": "★★★★☆", "recommended": True}], "benchmark": {"mmlu": 68.3, "humaneval": 62.8, "gsm8k": 80.5, "avg_vram_q4km_gb": 3.0, "speed_tok_s": 80}, "min_vram_gb": 3.5, "recommended_for": ["Chat", "RAG / QA", "Summarization"]},
]


# ═══════════════════════════════════════════════════════════════════════
#  INFERENCE ENGINE 
# ═══════════════════════════════════════════════════════════════════════
class InferenceEngine:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.llm = None
            cls._instance.current_model_id = None
        return cls._instance

    def load_model(self, model_id: str, model_path: str, config: AppConfig):
        if not HAS_LLAMA_CPP: raise Exception("llama-cpp-python is not installed.")
        if self.current_model_id == model_id and self.llm is not None: return 
        self.unload()
        self.llm = LlamaCPP(model_path=model_path, n_ctx=config.n_ctx, n_gpu_layers=config.n_gpu_layers, n_threads=config.n_threads, n_batch=config.n_batch, verbose=False)
        self.current_model_id = model_id

    def unload(self):
        if self.llm is not None: del self.llm; self.llm = None; self.current_model_id = None

    def chat_stream(self, messages: list, config: AppConfig):
        if self.llm is None: raise Exception("No model loaded.")
        mirostat_mode = config.mirostat_mode
        top_p = config.top_p if mirostat_mode == 0 else 1.0
        min_p = config.min_p if mirostat_mode == 0 else 1.0
        response = self.llm.create_chat_completion(
            messages=messages, max_tokens=config.max_tokens, temperature=config.temp,
            top_p=top_p, min_p=min_p, frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty, mirostat_mode=mirostat_mode,
            mirostat_tau=config.mirostat_tau, mirostat_eta=config.mirostat_eta, stream=True
        )
        for chunk in response:
            delta = chunk["choices"][0].get("delta", {}); content = delta.get("content", "")
            if content: yield content


# ═══════════════════════════════════════════════════════════════════════
#  MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════
class ModelRegistry:
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir); self.models_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.models_dir / "manifest.json"; self._manifest = self._load()

    def _load(self) -> dict:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path) as f: return json.load(f)
            except Exception: pass
        return {"models": {}}

    def save(self):
        with open(self.manifest_path, "w") as f: json.dump(self._manifest, f, indent=2)

    @property
    def models(self) -> dict: return self._manifest["models"]

    def add(self, model_id: str, info: dict): self._manifest["models"][model_id] = info; self.save()

    def remove(self, model_id: str):
        if model_id in self._manifest["models"]:
            info = self._manifest["models"][model_id]; model_path = Path(info.get("path", ""))
            if model_path.is_symlink(): model_path.unlink()
            elif model_path.exists():
                if model_path.is_file(): model_path.unlink()
                elif model_path.is_dir(): shutil.rmtree(model_path)
            cold_path = info.get("cold_path")
            if cold_path and Path(cold_path).exists(): Path(cold_path).unlink()
            del self._manifest["models"][model_id]; self.save()

    def get(self, model_id: str) -> dict: return self._manifest["models"].get(model_id, {})

    def list_models(self) -> list:
        result = []
        for mid, info in self._manifest["models"].items():
            p = Path(info.get("path", ""))
            if p.exists():
                actual_size = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                info["_actual_size"] = actual_size
            result.append((mid, info))
        return result

    def scan_for_orphaned(self) -> list:
        tracked_paths = {info.get("path", "") for info in self._manifest["models"].values()}
        orphans = []
        for f in self.models_dir.rglob("*.gguf"):
            if str(f) not in tracked_paths and not f.name.endswith(".part"): orphans.append(str(f))
        for f in self.models_dir.rglob("*.part"): orphans.append(str(f))
        return orphans

    def import_model(self, filepath: str, model_id: str = None) -> str:
        src = Path(filepath)
        if not model_id: model_id = src.stem
        dest = self.models_dir / f"{model_id}.gguf"
        if src.resolve() != dest.resolve(): shutil.copy2(src, dest)
        else: dest = src
        self.add(model_id, {"path": str(dest), "source": "local_import", "filename": dest.name, "size": dest.stat().st_size, "imported_at": datetime.now().isoformat(), "tags": [], "storage_tier": "hot"})
        return model_id


# ═══════════════════════════════════════════════════════════════════════
#  WORKER THREADS
# ═══════════════════════════════════════════════════════════════════════
class SearchWorker(QThread):
    results_ready = Signal(list, str); error = Signal(str)
    def __init__(self, query: str, source: str = "huggingface", limit: int = 30): super().__init__(); self.query, self.source, self.limit = query, source, limit
    def run(self):
        try:
            if self.source == "huggingface": self._search_hf()
            elif self.source == "modelscope": self._search_ms()
            elif self.source == "github": self._search_gh()
        except Exception as e: self.error.emit(str(e))
    def _search_hf(self):
        params = {"search": self.query, "filter": "gguf", "sort": "downloads", "direction": -1, "limit": self.limit}
        r = requests.get(f"{HF_API}/models", params=params, timeout=15); r.raise_for_status()
        results = [{"id": m.get("id",""), "downloads": m.get("downloads",0), "likes": m.get("likes",0), "source": "huggingface", "url": f"https://huggingface.co/{m.get('id','')}"} for m in r.json()]
        self.results_ready.emit(results, "huggingface")
    def _search_ms(self):
        params = {"Name": self.query, "PageSize": self.limit, "SortBy": "GmtDownload"}
        r = requests.get(f"{MS_API}/models", params=params, timeout=15); r.raise_for_status()
        data = r.json(); models = data.get("Data", data.get("Models", []))
        if isinstance(models, dict): models = models.get("Models", [])
        results = [{"id": m.get("Id", m.get("Name","")), "downloads": m.get("DownloadCount",0), "source": "modelscope", "url": f"https://modelscope.cn/models/{m.get('Id','')}"} for m in models]
        self.results_ready.emit(results, "modelscope")
    def _search_gh(self):
        params = {"q": f"{self.query} gguf llama model", "per_page": self.limit}
        r = requests.get(f"{GH_API}/search/repositories", params=params, timeout=15); r.raise_for_status()
        results = [{"id": m.get("full_name",""), "downloads": m.get("stargazers_count",0), "source": "github", "url": m.get("html_url","")} for m in r.json().get("items",[])]
        self.results_ready.emit(results, "github")

class ModelFilesWorker(QThread):
    files_ready = Signal(list, str); error = Signal(str)
    def __init__(self, model_id: str, source: str = "huggingface"): super().__init__(); self.model_id, self.source = model_id, source
    def run(self):
        try:
            if self.source == "huggingface": self._hf_files()
            elif self.source == "modelscope": self._ms_files()
            elif self.source == "github": self._gh_files()
        except Exception as e: self.error.emit(str(e))
    def _hf_files(self):
        r = requests.get(f"{HF_API}/models/{self.model_id}", timeout=15); r.raise_for_status()
        files = [{"name": s.get("rfilename",""), "size": s.get("size",0), "url": f"https://huggingface.co/{self.model_id}/resolve/main/{s.get('rfilename','')}"} for s in r.json().get("siblings",[]) if s.get("rfilename","").endswith(".gguf")]
        self.files_ready.emit(files, "huggingface")
    def _ms_files(self):
        r = requests.get(f"{MS_API}/models/{self.model_id}/repo/files", params={"Revision": "master"}, timeout=15); r.raise_for_status()
        files = [{"name": f.get("Name", f.get("name","")), "size": f.get("Size", f.get("size",0)), "url": f"https://modelscope.cn/api/v1/models/{self.model_id}/repo?Revision=master&FilePath={f.get('Name','')}"} for f in r.json().get("Data",[]) if f.get("Name", f.get("name","")).endswith(".gguf")]
        self.files_ready.emit(files, "modelscope")
    def _gh_files(self):
        r = requests.get(f"{GH_API}/repos/{self.model_id}/releases", timeout=15); r.raise_for_status()
        files = [{"name": f"{rel.get('tag_name','')}/{a.get('name','')}", "size": a.get("size",0), "url": a.get("browser_download_url","")} for rel in r.json()[:3] for a in rel.get("assets",[]) if a.get("name","").endswith((".gguf",".bin"))]
        self.files_ready.emit(files, "github")

class ModelStatsWorker(QThread):
    stats_ready = Signal(dict); error = Signal(str)
    def __init__(self, repo_id: str): super().__init__(); self.repo_id = repo_id
    def run(self):
        try:
            r = requests.get(f"{HF_API}/models/{self.repo_id}", timeout=15); r.raise_for_status()
            data = r.json()
            stats = {"repo_id": self.repo_id, "downloads": data.get("downloads", 0), "likes": data.get("likes", 0), "tags": data.get("tags", []), "pipeline_tag": data.get("pipeline_tag", ""), "library_name": data.get("library_name", ""), "created_at": data.get("createdAt", ""), "last_modified": data.get("lastModified", ""), "gguf_files": [s.get("rfilename", "") for s in data.get("siblings", []) if s.get("rfilename", "").endswith(".gguf")]}
            try:
                readme_r = requests.get(f"https://huggingface.co/{self.repo_id}/raw/main/README.md", timeout=10)
                if readme_r.status_code == 200: stats["readme"] = readme_r.text[:5000]
            except Exception: pass
            self.stats_ready.emit(stats)
        except Exception as e: self.error.emit(str(e))

class DownloadWorker(QThread):
    progress = Signal(int, int, float); finished = Signal(bool, str)
    def __init__(self, url: str, dest_path: str, model_id: str, source: str, registry: ModelRegistry, filename: str): super().__init__(); self.url, self.dest_path, self.model_id, self.source, self.registry, self.filename = url, dest_path, model_id, source, registry, filename; self._cancel = False
    def cancel(self): self._cancel = True
    def run(self):
        tmp_path = self.dest_path + ".part"
        try:
            Path(self.dest_path).parent.mkdir(parents=True, exist_ok=True); headers, downloaded = {}, 0
            if Path(tmp_path).exists(): downloaded = Path(tmp_path).stat().st_size; headers["Range"] = f"bytes={downloaded}-"
            r = requests.get(self.url, stream=True, headers=headers, timeout=30); total = int(r.headers.get("content-length", 0))
            if downloaded > 0 and r.status_code == 206: total += downloaded
            elif r.status_code == 200: downloaded, total = 0, int(r.headers.get("content-length", 0))
            mode = "ab" if downloaded > 0 else "wb"; last_time, last_bytes, speed = time.time(), downloaded, 0.0
            with open(tmp_path, mode) as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if self._cancel: r.close()
                    if Path(tmp_path).exists(): Path(tmp_path).unlink()
                    self.finished.emit(False, "Cancelled"); return
                    f.write(chunk); downloaded += len(chunk); now = time.time(); elapsed = now - last_time
                    if elapsed >= 0.5: speed = (downloaded - last_bytes) / elapsed / 1024; last_time, last_bytes = now, downloaded
                    self.progress.emit(downloaded, total, speed)
            if Path(tmp_path).exists(): Path(tmp_path).rename(self.dest_path)
            sha256 = hashlib.sha256()
            with open(self.dest_path, "rb") as f: 
                for byte_block in iter(lambda: f.read(4096), b""): sha256.update(byte_block)
            self.registry.add(self.model_id, {"path": self.dest_path, "source": self.source, "filename": self.filename, "size": Path(self.dest_path).stat().st_size, "url": self.url, "sha256": sha256.hexdigest(), "downloaded_at": datetime.now().isoformat(), "tags": [], "storage_tier": "hot"})
            self.finished.emit(True, f"Downloaded {self.model_id}")
        except Exception as e:
            if Path(tmp_path).exists() and self._cancel: Path(tmp_path).unlink()
            self.finished.emit(False, str(e))

class InferenceWorker(QThread):
    token_generated = Signal(str); finished_signal = Signal(); error = Signal(str)
    def __init__(self, engine: InferenceEngine, messages: list, config: AppConfig): super().__init__(); self.engine, self.messages, self.config = engine, messages, config; self._cancel = False
    def cancel(self): self._cancel = True
    def run(self):
        try:
            for token in self.engine.chat_stream(self.messages, self.config):
                if self._cancel: break
                self.token_generated.emit(token)
            self.finished_signal.emit()
        except Exception as e: self.error.emit(str(e))

class LoadModelWorker(QThread):
    loaded = Signal(str); error = Signal(str)
    def __init__(self, engine: InferenceEngine, model_id: str, model_path: str, config: AppConfig): super().__init__(); self.engine, self.model_id, self.model_path, self.config = engine, model_id, model_path, config
    def run(self):
        try:
            self.engine.load_model(self.model_id, self.model_path, self.config)
            self.loaded.emit(self.model_id)
        except Exception as e: self.error.emit(str(e))

class SystemMonitor(QThread):
    stats_updated = Signal(dict)
    def __init__(self): super().__init__(); self._running = True
    def stop(self): self._running = False
    def run(self):
        while self._running:
            mem = psutil.virtual_memory(); cpu = psutil.cpu_percent(interval=0.5); gpus = GPUDetector.detect_gpus()
            self.stats_updated.emit({"cpu_percent": cpu, "ram_total": mem.total, "ram_used": mem.used, "ram_available": mem.available, "ram_percent": mem.percent, "gpu": [{"name": g["name"], "total_mb": g["total_vram_mb"], "used_mb": g["total_vram_mb"] - g["free_vram_mb"], "free_mb": g["free_vram_mb"]} for g in gpus]})
            time.sleep(2)


# ═══════════════════════════════════════════════════════════════════════
#  FASTAPI SERVER
# ═══════════════════════════════════════════════════════════════════════
if HAS_FASTAPI:
    api_app = FastAPI(title="LocalAI Studio Pro API")

    class ChatMessage(PydanticModel):
        role: str; content: str

    class ChatCompletionRequest(PydanticModel):
        model: str = "default"
        messages: List[ChatMessage]
        temperature: float = 0.7
        max_tokens: int = 2048
        stream: bool = False
        min_p: Optional[float] = None
        frequency_penalty: Optional[float] = None
        presence_penalty: Optional[float] = None
        mirostat_mode: Optional[int] = None
        mirostat_tau: Optional[float] = None
        mirostat_eta: Optional[float] = None

    @api_app.get("/v1/models")
    async def list_models():
        config = AppConfig(); registry = ModelRegistry(config.models_dir); data = []
        for mid, info in registry.list_models(): data.append({"id": mid, "object": "model", "owned_by": info.get("source", "local")})
        return {"object": "list", "data": data}

    @api_app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        config = AppConfig(); engine = InferenceEngine(); registry = ModelRegistry(config.models_dir)
        if req.model != engine.current_model_id:
            model_info = registry.get(req.model)
            if not model_info or not Path(model_info.get("path","")).exists(): raise HTTPException(status_code=404, detail=f"Model {req.model} not found.")
            try: engine.load_model(req.model, model_info["path"], config)
            except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        if req.temperature is not None: config.set("temperature", req.temperature)
        if req.max_tokens is not None: config.set("max_tokens", req.max_tokens)
        if req.min_p is not None: config.set("min_p", req.min_p)
        messages_dicts = [m.dict() for m in req.messages]
        if req.stream:
            def generate():
                for token in engine.chat_stream(messages_dicts, config):
                    chunk = {"choices": [{"delta": {"content": token}}]}; yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            full_text = ""
            for token in engine.chat_stream(messages_dicts, config): full_text += token
            return {"object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": full_text}}], "model": req.model}

class ServerWorker(QThread):
    log_message = Signal(str); started_signal = Signal(); stopped_signal = Signal()
    def __init__(self, port: int, cors: bool): super().__init__(); self.port, self.cors = port, cors; self._server = None
    def run(self):
        if not HAS_FASTAPI: self.log_message.emit("ERROR: FastAPI not installed."); return
        if self.cors: api_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
        config = uvicorn.Config(api_app, host="0.0.0.0", port=self.port, log_level="info")
        self._server = uvicorn.Server(config)
        self.log_message.emit(f"Server starting on http://localhost:{self.port}"); self.started_signal.emit()
        try: import asyncio; asyncio.run(self._server.serve())
        except Exception as e: self.log_message.emit(f"Server error: {str(e)}")
        finally: self.stopped_signal.emit()
    def stop(self):
        if self._server: self._server.should_exit = True


# ═══════════════════════════════════════════════════════════════════════
#  CUSTOM WIDGETS
# ═══════════════════════════════════════════════════════════════════════
class PieChart(QWidget):
    def __init__(self, parent=None): super().__init__(parent); self.setFixedSize(160, 160); self.data = []; self.bg_color = DARK_THEME['bg']
    def set_data(self, data, bg_color=None): 
        self.data = data
        if bg_color: self.bg_color = bg_color
        self.update()
    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); size = min(self.width(), self.height()) - 20
        rect = QRect(10, 10, size, size); total = sum(d[1] for d in self.data if d[1] > 0)
        if total == 0: p.end(); return
        start_angle = 0
        for label, value, color in self.data:
            if value <= 0: continue
            span = int(value / total * 360 * 16)
            p.setBrush(QBrush(color)); p.setPen(QPen(QColor(self.bg_color), 2))
            p.drawPie(rect, start_angle, span); start_angle += span
        p.setBrush(QBrush(QColor(self.bg_color))); p.setPen(Qt.NoPen)
        p.drawEllipse(rect.center(), size//4, size//4); p.end()

class ChatBubble(QFrame):
    regenerate_requested = Signal()
    def __init__(self, text: str, role: str = "user", colors: dict = None, parent=None):
        super().__init__(parent); self.role = role; self._raw_text = text; c = colors or DARK_THEME
        lay = QVBoxLayout(self); lay.setContentsMargins(12, 8, 12, 8); header_lay = QHBoxLayout()
        role_label = QLabel(role.upper()); role_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        role_label.setStyleSheet(f"color: {c['accent'] if role=='user' else c['primary']}; border: none;")
        header_lay.addWidget(role_label); header_lay.addStretch()
        self.copy_btn = QToolButton(); self.copy_btn.setText("📋"); self.copy_btn.setFixedSize(24, 24)
        self.copy_btn.setStyleSheet(f"QToolButton {{ background: transparent; border: none; }} QToolButton:hover {{ background: {c['secondary_hover']}; border-radius: 4px; }}")
        self.copy_btn.setToolTip("Copy Text"); self.copy_btn.clicked.connect(self._copy_text); self.copy_btn.setVisible(False)
        header_lay.addWidget(self.copy_btn)
        lay.addLayout(header_lay)
        self.content_box = QTextBrowser(); self.content_box.setOpenExternalLinks(True)
        self.content_box.setStyleSheet(f"QTextBrowser {{ background: transparent; border: none; color: {c['text']}; padding: 4px; }}")
        self.content_box.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff); self.content_box.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff); self.content_box.setReadOnly(True)
        self._render_markdown(text); lay.addWidget(self.content_box)
        bg = c['user_bubble'] if role == "user" else c['bg']
        self.setStyleSheet(f"ChatBubble {{ background: {bg}; border: 1px solid {c['border']}; border-radius: 10px; margin: 4px 0; }}")

    def _render_markdown(self, text):
        if HAS_MARKDOWN: html = markdown.markdown(text, extensions=['fenced_code', 'codehilite']); self.content_box.setHtml(html)
        else: self.content_box.setPlainText(text)
        self.content_box.document().setTextWidth(self.content_box.width() if self.content_box.width() > 50 else 600)
        self.content_box.setFixedHeight(int(self.content_box.document().size().height()) + 10)

    def append_text(self, text): self._raw_text += text; self._render_markdown(self._raw_text)
    def _copy_text(self): QApplication.clipboard().setText(self._raw_text)
    def enterEvent(self, event): self.copy_btn.setVisible(True); super().enterEvent(event)
    def leaveEvent(self, event): self.copy_btn.setVisible(False); super().leaveEvent(event)


# ═══════════════════════════════════════════════════════════════════════
#  PAGES
# ═══════════════════════════════════════════════════════════════════════

class DashboardPage(QWidget):
    def __init__(self, config: AppConfig, registry: ModelRegistry, parent=None):
        super().__init__(parent); self.config, self.registry = config, registry; self._build()
    def _build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(30,20,30,20); lay.setSpacing(16)
        title = QLabel("📊 Dashboard & Hardware"); title.setFont(QFont("Segoe UI", 22, QFont.Bold)); lay.addWidget(title)
        charts_lay = QHBoxLayout()
        ram_grp = QGroupBox("System RAM"); ram_lay = QVBoxLayout(ram_grp)
        self.ram_label = QLabel("RAM: --"); self.ram_pie = PieChart(); ram_lay.addWidget(self.ram_label); ram_lay.addWidget(self.ram_pie, 0, Qt.AlignCenter)
        charts_lay.addWidget(ram_grp)
        vram_grp = QGroupBox("GPU VRAM"); vram_lay = QVBoxLayout(vram_grp)
        self.vram_label = QLabel("GPU: No NVIDIA GPU detected"); self.vram_pie = PieChart(); vram_lay.addWidget(self.vram_label); vram_lay.addWidget(self.vram_pie, 0, Qt.AlignCenter)
        charts_lay.addWidget(vram_grp); lay.addLayout(charts_lay)
        stats_grp = QGroupBox("Model Statistics"); stats_lay = QFormLayout(stats_grp)
        self.models_count_label = QLabel("0"); self.total_size_label = QLabel("0 B")
        stats_lay.addRow("Installed Models:", self.models_count_label); stats_lay.addRow("Total Size:", self.total_size_label)
        lay.addWidget(stats_grp); lay.addStretch()

    def update_stats(self, sys_info: dict = None):
        models = self.registry.list_models(); self.models_count_label.setText(str(len(models)))
        total = sum(info.get("_actual_size", info.get("size", 0)) for _, info in models); self.total_size_label.setText(fmt_size(total))
        c = ThemeManager().get_theme(self.config.theme_name)
        if sys_info:
            ram_t, ram_u, ram_a = sys_info["ram_total"], sys_info["ram_used"], sys_info["ram_available"]
            self.ram_label.setText(f"RAM: {fmt_size(ram_u)} / {fmt_size(ram_t)}")
            self.ram_pie.set_data([("Used", ram_u, QColor(c['accent'])), ("Available", ram_a, QColor(c['primary']))], bg_color=c['bg'])
            gpus = sys_info.get("gpu", [])
            if gpus:
                g = gpus[0]; self.vram_label.setText(f"VRAM: {fmt_size(g['used_mb']*1e6)} / {fmt_size(g['total_mb']*1e6)}")
                self.vram_pie.set_data([("Used", g['used_mb'], QColor(c['accent'])), ("Free", g['free_mb'], QColor(c['primary']))], bg_color=c['bg'])
            else: self.vram_label.setText("GPU: No NVIDIA GPU detected"); self.vram_pie.set_data([], bg_color=c['bg'])

class ModelInfoCard(QFrame):
    load_requested = Signal(str); delete_requested = Signal(str)
    def __init__(self, model_id: str, info: dict, colors: dict = None, parent=None):
        super().__init__(parent); self.model_id = model_id; self.info = info; c = colors or DARK_THEME
        self.setStyleSheet(f"ModelInfoCard {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 10px; margin: 4px 0; }} ModelInfoCard:hover {{ border-color: {c['accent']}; }}")
        lay = QVBoxLayout(self); lay.setContentsMargins(16, 12, 16, 12); lay.setSpacing(8)
        header_lay = QHBoxLayout()
        name_lbl = QLabel(f"<b>{model_id}</b>"); name_lbl.setFont(QFont("Segoe UI", 13, QFont.Bold)); header_lay.addWidget(name_lbl)
        tier = info.get("storage_tier", "hot"); tier_lbl = QLabel("🔥 Hot" if tier == "hot" else "❄️ Cold")
        tier_lbl.setStyleSheet(f"font-size: 11px; padding: 2px 8px; border-radius: 4px; background: {c['primary'] if tier == 'hot' else c['accent']}; color: white;"); header_lay.addWidget(tier_lbl)
        src_lbl = QLabel(info.get("source", "local").upper()); src_lbl.setStyleSheet(f"font-size: 10px; color: {c['text_muted']};"); header_lay.addWidget(src_lbl); header_lay.addStretch()
        self.load_btn = QPushButton("▶ Load"); self.load_btn.setObjectName("primaryBtn"); self.load_btn.setFixedWidth(80); self.load_btn.clicked.connect(lambda: self.load_requested.emit(model_id)); header_lay.addWidget(self.load_btn)
        self.delete_btn = QPushButton("🗑"); self.delete_btn.setObjectName("dangerBtn"); self.delete_btn.setFixedWidth(36); self.delete_btn.clicked.connect(lambda: self.delete_requested.emit(model_id)); header_lay.addWidget(self.delete_btn)
        lay.addLayout(header_lay)
        details_lay = QHBoxLayout()
        size = info.get("_actual_size", info.get("size", 0)); details_lay.addWidget(self._detail_item("💾 Size", fmt_size(size)))
        details_lay.addWidget(self._detail_item("📁 File", info.get("filename", "N/A")))
        downloaded = info.get("downloaded_at", info.get("imported_at", "Unknown"))
        if downloaded and downloaded != "Unknown": downloaded = downloaded[:10]
        details_lay.addWidget(self._detail_item("📅 Added", downloaded)); details_lay.addStretch(); lay.addLayout(details_lay)

    def _detail_item(self, label: str, value: str) -> QWidget:
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(0, 0, 12, 0); l.setSpacing(1)
        lbl = QLabel(label); lbl.setStyleSheet("font-size: 10px; color: #8b949e;"); l.addWidget(lbl)
        val = QLabel(str(value)); val.setStyleSheet("font-size: 12px;"); val.setWordWrap(True); l.addWidget(val); return w

class InstalledModelsPage(QWidget):
    model_load_requested = Signal(str); model_delete_requested = Signal(str)
    def __init__(self, config: AppConfig, registry: ModelRegistry, parent=None):
        super().__init__(parent); self.config, self.registry = config, registry; self._build()
    def _build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(30,20,30,20); lay.setSpacing(12)
        header = QHBoxLayout(); title = QLabel("📦 Installed Models"); title.setFont(QFont("Segoe UI", 22, QFont.Bold))
        header.addWidget(title); header.addStretch()
        import_btn = QPushButton("📁 Import"); import_btn.setObjectName("primaryBtn"); import_btn.clicked.connect(self._import_model); header.addWidget(import_btn)
        refresh_btn = QPushButton("🔄 Refresh"); refresh_btn.setObjectName("secondaryBtn"); refresh_btn.clicked.connect(self.refresh); header.addWidget(refresh_btn)
        lay.addLayout(header)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll_widget = QWidget(); self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignTop); self.scroll.setWidget(self.scroll_widget)
        lay.addWidget(self.scroll)
        self.refresh()

    def refresh(self):
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0); widget = item.widget()
            if widget: widget.deleteLater()
        c = ThemeManager().get_theme(self.config.theme_name)
        for mid, info in self.registry.list_models():
            card = ModelInfoCard(mid, info, c); card.load_requested.connect(self.model_load_requested.emit); card.delete_requested.connect(self._delete_model)
            self.scroll_layout.addWidget(card)

    def _import_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import GGUF Model", "", "GGUF Files (*.gguf);;All Files (*)")
        if path:
            try: self.registry.import_model(path); self.refresh()
            except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def _delete_model(self, model_id: str):
        reply = QMessageBox.question(self, "Delete Model", f"Are you sure you want to delete {model_id}?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes: self.registry.remove(model_id); self.refresh()

class EnhancedModelBrowserPage(QWidget):
    download_requested = Signal(str, str, str, str)
    def __init__(self, config: AppConfig, registry: ModelRegistry, parent=None):
        super().__init__(parent); self.config = config; self.registry = registry; self.gpu_manager = SmartGPUManager(config); self._results = []; self._build()

    def _build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(30, 20, 30, 20); lay.setSpacing(12)
        title = QLabel("🔍 Model Browser & Catalog"); title.setFont(QFont("Segoe UI", 22, QFont.Bold)); lay.addWidget(title)
        tabs = QTabWidget()

        catalog_widget = QWidget(); catalog_lay = QVBoxLayout(catalog_widget)
        filter_lay = QHBoxLayout()
        self.family_filter = QComboBox(); self.family_filter.addItem("All Families")
        families = sorted(set(m["family"] for m in RECOMMENDED_MODELS)); self.family_filter.addItems(families); self.family_filter.currentTextChanged.connect(self._filter_catalog); filter_lay.addWidget(QLabel("Family:")); filter_lay.addWidget(self.family_filter)
        self.vram_filter = QCheckBox("Only models that fit my GPU"); self.vram_filter.stateChanged.connect(self._filter_catalog); filter_lay.addWidget(self.vram_filter); filter_lay.addStretch()
        catalog_lay.addLayout(filter_lay)
        self.catalog_table = QTableWidget(0, 5); self.catalog_table.setHorizontalHeaderLabels(["Model", "Params", "Best Variant", "Min VRAM", "Action"]); self.catalog_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch); self.catalog_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        catalog_lay.addWidget(self.catalog_table); tabs.addTab(catalog_widget, "📋 Recommended Catalog")

        search_widget = QWidget(); search_lay = QVBoxLayout(search_widget); search_lay.setContentsMargins(0, 8, 0, 0)
        search_bar = QHBoxLayout(); self.search_input = QLineEdit(); self.search_input.setPlaceholderText("Search HuggingFace, ModelScope, GitHub..."); self.search_input.returnPressed.connect(self._do_search)
        self.source_combo = QComboBox(); self.source_combo.addItems(["HuggingFace", "ModelScope", "GitHub"])
        self.search_btn = QPushButton("Search"); self.search_btn.setObjectName("primaryBtn"); self.search_btn.clicked.connect(self._do_search)
        search_bar.addWidget(self.search_input, 1); search_bar.addWidget(self.source_combo); search_bar.addWidget(self.search_btn); search_lay.addLayout(search_bar)
        self.results_list = QListWidget(); self.results_list.currentRowChanged.connect(self._on_result_selected); search_lay.addWidget(self.results_list)
        self.files_table = QTableWidget(0, 3); self.files_table.setHorizontalHeaderLabels(["File", "Size", "Action"]); self.files_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch); self.files_table.setEditTriggers(QAbstractItemView.NoEditTriggers); search_lay.addWidget(self.files_table)
        tabs.addTab(search_widget, "🔎 Search Models"); lay.addWidget(tabs)
        self._filter_catalog()

    def _filter_catalog(self):
        self.catalog_table.setRowCount(0); family = self.family_filter.currentText(); vram_only = self.vram_filter.isChecked()
        max_vram_gb = 999
        if vram_only and self.gpu_manager.gpus:
            best = max(self.gpu_manager.gpus, key=lambda g: g.get("free_vram_mb", 0)); max_vram_gb = best.get("free_vram_mb", 0) / 1024
        for model in RECOMMENDED_MODELS:
            if family != "All Families" and model["family"] != family: continue
            if model["min_vram_gb"] > max_vram_gb: continue
            rec_variant = next((v for v in model["variants"] if v.get("recommended")), model["variants"][0])
            row = self.catalog_table.rowCount(); self.catalog_table.insertRow(row)
            self.catalog_table.setItem(row, 0, QTableWidgetItem(f"{model['id']}\n{model.get('description', '')[:80]}..."))
            self.catalog_table.setItem(row, 1, QTableWidgetItem(f"{model['n_params_billion']:.1f}B"))
            self.catalog_table.setItem(row, 2, QTableWidgetItem(f"{rec_variant['name']} ({rec_variant['size_gb']:.1f}GB) {rec_variant['quality']}"))
            vram_item = QTableWidgetItem(f"{model['min_vram_gb']:.0f} GB")
            if model["min_vram_gb"] <= 6: vram_item.setForeground(QColor("#238636"))
            elif model["min_vram_gb"] <= 16: vram_item.setForeground(QColor("#9e6a03"))
            else: vram_item.setForeground(QColor("#da3633"))
            self.catalog_table.setItem(row, 3, vram_item)
            dl_btn = QPushButton("⬇ Download"); dl_btn.setObjectName("primaryBtn"); dl_btn.clicked.connect(lambda checked, m=model: self.download_requested.emit(f"https://huggingface.co/{m['repo_id']}/resolve/main/{next((v for v in m['variants'] if v.get('recommended')), m['variants'][0])['filename']}", m['id'], "huggingface", next((v for v in m['variants'] if v.get('recommended')), m['variants'][0])['filename']))
            self.catalog_table.setCellWidget(row, 4, dl_btn)

    def _do_search(self):
        q = self.search_input.text().strip()
        if not q: return
        src = self.source_combo.currentText().lower().replace(" ", "")
        self.search_btn.setText("Searching..."); self.search_btn.setEnabled(False)
        self._search_worker = SearchWorker(q, src); self._search_worker.results_ready.connect(self._on_results); self._search_worker.finished.connect(lambda: (self.search_btn.setText("Search"), self.search_btn.setEnabled(True))); self._search_worker.start()

    def _on_results(self, results, source):
        self._results = results; self.results_list.clear()
        for r in results: self.results_list.addItem(f"{r['id']}  ⬇ {r.get('downloads',0):,}")

    def _on_result_selected(self, row):
        if row < 0 or row >= len(self._results): return
        r = self._results[row]; self.files_table.setRowCount(0)
        self._files_worker = ModelFilesWorker(r["id"], r.get("source", "huggingface")); self._files_worker.files_ready.connect(self._on_files); self._files_worker.start()

    def _on_files(self, files, source):
        self.files_table.setRowCount(0)
        for f in files:
            row = self.files_table.rowCount(); self.files_table.insertRow(row); self.files_table.setItem(row, 0, QTableWidgetItem(f["name"])); self.files_table.setItem(row, 1, QTableWidgetItem(fmt_size(f.get("size", 0))))
            btn = QPushButton("⬇ Download"); btn.setObjectName("primaryBtn"); btn.clicked.connect(lambda checked, url=f["url"], name=f["name"], src=source: self.download_requested.emit(url, name.replace(".gguf", ""), src, name)); self.files_table.setCellWidget(row, 2, btn)


class GPUSettingsPage(QWidget):
    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent); self.config = config; self.gpu_manager = SmartGPUManager(config); self._build()

    def _build(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True); container = QWidget(); lay = QVBoxLayout(container); lay.setContentsMargins(30, 20, 30, 20); lay.setSpacing(16)
        title = QLabel("🖥️ GPU & Hardware Settings"); title.setFont(QFont("Segoe UI", 22, QFont.Bold)); lay.addWidget(title)
        detect_grp = QGroupBox("Detected GPUs"); detect_lay = QVBoxLayout(detect_grp)
        self.gpu_table = QTableWidget(0, 4); self.gpu_table.setHorizontalHeaderLabels(["GPU", "VRAM Total", "VRAM Free", "Type"]); self.gpu_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch); self.gpu_table.setEditTriggers(QAbstractItemView.NoEditTriggers); self.gpu_table.setMaximumHeight(200); detect_lay.addWidget(self.gpu_table)
        detect_btn = QPushButton("🔄 Re-detect GPUs"); detect_btn.setObjectName("secondaryBtn"); detect_btn.clicked.connect(self._detect_gpus); detect_lay.addWidget(detect_btn); lay.addWidget(detect_grp)
        smart_grp = QGroupBox("Smart GPU Offloading"); smart_lay = QFormLayout(smart_grp)
        self.auto_gpu_check = QCheckBox("Auto-calculate optimal GPU layers on model load"); self.auto_gpu_check.setChecked(self.config.auto_gpu_layers); smart_lay.addRow(self.auto_gpu_check)
        self.vram_headroom_slider = QSlider(Qt.Horizontal); self.vram_headroom_slider.setRange(5, 30); self.vram_headroom_slider.setValue(int(self.config.vram_headroom_pct)); self.headroom_label = QLabel(f"{self.vram_headroom_slider.value()}%"); self.vram_headroom_slider.valueChanged.connect(lambda v: self.headroom_label.setText(f"{v}%"))
        headroom_lay = QHBoxLayout(); headroom_lay.addWidget(self.vram_headroom_slider); headroom_lay.addWidget(self.headroom_label); smart_lay.addRow("VRAM Headroom:", headroom_lay)
        lay.addWidget(smart_grp)
        manual_grp = QGroupBox("Manual GPU Configuration"); manual_lay = QFormLayout(manual_grp)
        self.n_gpu_spin = QSpinBox(); self.n_gpu_spin.setRange(-1, 999); self.n_gpu_spin.setValue(self.config.n_gpu_layers); self.n_gpu_spin.setSpecialValueText("All Layers (-1)"); manual_lay.addRow("GPU Layers (-1=all):", self.n_gpu_spin)
        self.n_ctx_spin = QSpinBox(); self.n_ctx_spin.setRange(512, 131072); self.n_ctx_spin.setSingleStep(512); self.n_ctx_spin.setValue(self.config.n_ctx); manual_lay.addRow("Context Size:", self.n_ctx_spin)
        self.n_threads_spin = QSpinBox(); self.n_threads_spin.setRange(1, 128); self.n_threads_spin.setValue(self.config.n_threads); manual_lay.addRow("CPU Threads:", self.n_threads_spin)
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setRange(1, 4096); self.batch_size_spin.setValue(self.config.n_batch); manual_lay.addRow("Batch Size:", self.batch_size_spin)
        lay.addWidget(manual_grp)
        save_btn = QPushButton("💾 Save Settings"); save_btn.setObjectName("primaryBtn"); save_btn.clicked.connect(self._save_settings); lay.addWidget(save_btn)
        lay.addStretch(); scroll.setWidget(container); outer = QVBoxLayout(self); outer.addWidget(scroll); self._detect_gpus()

    def _detect_gpus(self):
        self.gpu_manager.refresh(); self.gpu_table.setRowCount(0)
        for gpu in self.gpu_manager.gpus:
            row = self.gpu_table.rowCount(); self.gpu_table.insertRow(row)
            self.gpu_table.setItem(row, 0, QTableWidgetItem(gpu.get("name", "Unknown"))); self.gpu_table.setItem(row, 1, QTableWidgetItem(fmt_size(int(gpu.get("total_vram_mb", 0) * 1e6)))); self.gpu_table.setItem(row, 2, QTableWidgetItem(fmt_size(int(gpu.get("free_vram_mb", 0) * 1e6)))); self.gpu_table.setItem(row, 3, QTableWidgetItem(gpu.get("type", "unknown")))

    def _save_settings(self):
        self.config.set("n_gpu_layers", self.n_gpu_spin.value()); self.config.set("n_ctx", self.n_ctx_spin.value()); self.config.set("n_threads", self.n_threads_spin.value()); self.config.set("n_batch", self.batch_size_spin.value()); self.config.set("auto_gpu_layers", self.auto_gpu_check.isChecked()); self.config.set("vram_headroom_pct", self.vram_headroom_slider.value())
        QMessageBox.information(self, "Saved", "GPU settings saved successfully!")

class ModeSwitcherWidget(QFrame):
    mode_changed = Signal(str)
    def __init__(self, mode_manager: ModeManager, config: AppConfig, parent=None):
        super().__init__(parent); self.mode_manager = mode_manager; self.config = config; self.setObjectName("modeSwitcher"); self._build()

    def _build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(8, 4, 8, 4); lay.setSpacing(6)
        lbl = QLabel("⚡ Active Mode"); lbl.setFont(QFont("Segoe UI", 10, QFont.Bold)); lay.addWidget(lbl)
        self.mode_combo = QComboBox(); self._refresh_modes(); self.mode_combo.currentTextChanged.connect(self._on_mode_changed); lay.addWidget(self.mode_combo)
        self.info_label = QLabel(""); self.info_label.setWordWrap(True); self.info_label.setStyleSheet("font-size: 11px; color: #8b949e;"); lay.addWidget(self.info_label)
        self._update_info()

    def _refresh_modes(self):
        self.mode_combo.blockSignals(True); current = self.mode_combo.currentText(); self.mode_combo.clear()
        icons = {"Chat": "💬", "Code": "💻", "Creative Writing": "✍️", "RAG / QA": "📄", "Summarization": "📝"}
        for name, _, _ in self.mode_manager.list_modes(): self.mode_combo.addItem(f"{icons.get(name, '🔧')} {name}")
        idx = self.mode_combo.findText(current, Qt.MatchContains)
        if idx >= 0: self.mode_combo.setCurrentIndex(idx)
        self.mode_combo.blockSignals(False)

    def _on_mode_changed(self, text):
        name = text.split(" ", 1)[-1] if " " in text else text; self.mode_changed.emit(name); self._update_info()

    def _update_info(self):
        name = self.current_mode_name(); data = self.mode_manager.get_mode(name)
        self.info_label.setText(f"Temp: {data.get('temperature', 0.7)} | Ctx: {data.get('n_ctx', 4096):,}")

    def current_mode_name(self) -> str:
        text = self.mode_combo.currentText(); return text.split(" ", 1)[-1] if " " in text else text


class ChatPage(QWidget):
    def __init__(self, config: AppConfig, registry: ModelRegistry, engine: InferenceEngine, parent=None):
        super().__init__(parent); self.config = config; self.registry = registry; self.engine = engine; self._history = []; self._worker = None; self._build()

    def _build(self):
        splitter = QSplitter(Qt.Horizontal)
        # Chat Area
        chat_widget = QWidget(); chat_lay = QVBoxLayout(chat_widget); chat_lay.setContentsMargins(0,0,0,0)
        self.chat_scroll = QScrollArea(); self.chat_scroll.setWidgetResizable(True)
        self.chat_container = QWidget(); self.chat_layout = QVBoxLayout(self.chat_container); self.chat_layout.setAlignment(Qt.AlignTop); self.chat_scroll.setWidget(self.chat_container)
        chat_lay.addWidget(self.chat_scroll)
        input_lay = QHBoxLayout()
        self.input_edit = QPlainTextEdit(); self.input_edit.setMaximumHeight(100); self.input_edit.setPlaceholderText("Type your message...")
        self.send_btn = QPushButton("▶ Send"); self.send_btn.setObjectName("primaryBtn"); self.send_btn.clicked.connect(self._send_message)
        self.stop_btn = QPushButton("⏹ Stop"); self.stop_btn.setObjectName("dangerBtn"); self.stop_btn.clicked.connect(self._stop_inference); self.stop_btn.setEnabled(False)
        input_lay.addWidget(self.input_edit, 1); input_lay.addWidget(self.send_btn); input_lay.addWidget(self.stop_btn)
        chat_lay.addLayout(input_lay)
        splitter.addWidget(chat_widget)
        # Settings
        settings_widget = QWidget(); settings_lay = QVBoxLayout(settings_widget)
        mode_grp = QGroupBox("Model"); mode_lay = QFormLayout(mode_grp)
        self.model_combo = QComboBox(); self._refresh_models(); mode_lay.addRow("Active Model:", self.model_combo)
        load_btn = QPushButton("Load Model"); load_btn.setObjectName("primaryBtn"); load_btn.clicked.connect(self._load_selected_model); mode_lay.addRow(load_btn)
        settings_lay.addWidget(mode_grp)
        self.sys_prompt_edit = QPlainTextEdit(); self.sys_prompt_edit.setPlaceholderText("System prompt..."); self.sys_prompt_edit.setMaximumHeight(80)
        settings_lay.addWidget(QLabel("System Prompt:")); settings_lay.addWidget(self.sys_prompt_edit)
        params_grp = QGroupBox("Sampling"); params_lay = QFormLayout(params_grp)
        self.temp_spin = QDoubleSpinBox(); self.temp_spin.setRange(0.0, 2.0); self.temp_spin.setSingleStep(0.05); self.temp_spin.setValue(self.config.temp); params_lay.addRow("Temperature:", self.temp_spin)
        self.max_tokens_spin = QSpinBox(); self.max_tokens_spin.setRange(64, 65536); self.max_tokens_spin.setSingleStep(256); self.max_tokens_spin.setValue(self.config.max_tokens); params_lay.addRow("Max Tokens:", self.max_tokens_spin)
        settings_lay.addWidget(params_grp); settings_lay.addStretch()
        splitter.addWidget(settings_widget); splitter.setSizes([700, 300])
        main_lay = QVBoxLayout(self); main_lay.addWidget(splitter)

    def _refresh_models(self):
        self.model_combo.clear()
        for mid, _ in self.registry.list_models(): self.model_combo.addItem(mid)

    def _load_selected_model(self):
        mid = self.model_combo.currentText()
        if not mid: return
        info = self.registry.get(mid)
        if not info or not Path(info.get("path", "")).exists(): QMessageBox.warning(self, "Error", "Model path not found."); return
        
        if self.config.auto_gpu_layers:
            strategy = SmartGPUManager(self.config).get_optimal_gpu_layers(info["path"])
            self.config.set("n_gpu_layers", strategy.get("n_gpu_layers", 0))

        self._load_worker = LoadModelWorker(self.engine, mid, info["path"], self.config)
        self._load_worker.loaded.connect(lambda m: (QMessageBox.information(self, "Loaded", f"Model {m} loaded."), self._refresh_models()))
        self._load_worker.error.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self._load_worker.start()

    def _send_message(self):
        text = self.input_edit.toPlainText().strip()
        if not text: return
        c = ThemeManager().get_theme(self.config.theme_name)
        self.chat_layout.addWidget(ChatBubble(text, "user", c)); self._history.append({"role": "user", "content": text})
        self.input_edit.clear()
        
        assistant_bubble = ChatBubble("", "assistant", c); self.chat_layout.addWidget(assistant_bubble)
        messages = []
        sys = self.sys_prompt_edit.toPlainText().strip()
        if sys: messages.append({"role": "system", "content": sys})
        messages.extend(self._history)
        
        self.config.set("temperature", self.temp_spin.value()); self.config.set("max_tokens", self.max_tokens_spin.value())
        self._worker = InferenceWorker(self.engine, messages, self.config)
        self._worker.token_generated.connect(assistant_bubble.append_text)
        self._worker.finished_signal.connect(lambda: (self._history.append({"role": "assistant", "content": assistant_bubble._raw_text}), self.send_btn.setEnabled(True), self.stop_btn.setEnabled(False)))
        self._worker.error.connect(lambda e: (assistant_bubble.append_text(f"\n\n❌ Error: {e}"), self.send_btn.setEnabled(True), self.stop_btn.setEnabled(False)))
        self._worker.start(); self.send_btn.setEnabled(False); self.stop_btn.setEnabled(True)

    def _stop_inference(self):
        if self._worker: self._worker.cancel(); self.stop_btn.setEnabled(False)

    def load_mode(self, mode_name: str):
        mode_manager = ModeManager()
        model_id, sys_prompt, _ = mode_manager.apply_mode_to_config(mode_name, self.config)
        self.sys_prompt_edit.setPlainText(sys_prompt)
        self.temp_spin.setValue(self.config.temp); self.max_tokens_spin.setValue(self.config.max_tokens)
        if model_id:
            idx = self.model_combo.findText(model_id)
            if idx >= 0: self.model_combo.setCurrentIndex(idx)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME); self.resize(1200, 800)
        self.config = AppConfig(); self.registry = ModelRegistry(self.config.models_dir)
        self.engine = InferenceEngine(); self.mode_manager = ModeManager()
        self.theme_mgr = ThemeManager(); self.preset_mgr = PresetManager()
        
        self._build_ui(); self._apply_theme(); self._start_monitor()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_lay = QHBoxLayout(central); main_lay.setContentsMargins(0,0,0,0); main_lay.setSpacing(0)
        
        # Sidebar
        sidebar = QFrame(); sidebar.setObjectName("sidebar"); sidebar.setFixedWidth(240)
        side_lay = QVBoxLayout(sidebar); side_lay.setContentsMargins(8, 16, 8, 16); side_lay.setSpacing(4)
        
        logo = QLabel(f"<h2 style='color: #58a6ff;'>{APP_NAME}</h2><p style='color: #8b949e; font-size: 11px;'>v{APP_VERSION}</p>")
        side_lay.addWidget(logo); side_lay.addSpacing(10)
        
        self.mode_switcher = ModeSwitcherWidget(self.mode_manager, self.config)
        self.mode_switcher.mode_changed.connect(self._apply_mode)
        side_lay.addWidget(self.mode_switcher); side_lay.addSpacing(10)

        nav_buttons = ["📊 Dashboard", "💬 Chat", "📦 Installed Models", "🔍 Model Browser", "🖥️ GPU Settings"]
        self.stacked = QStackedWidget()
        self.nav_btns = []
        
        pages = [
            DashboardPage(self.config, self.registry),
            ChatPage(self.config, self.registry, self.engine),
            InstalledModelsPage(self.config, self.registry),
            EnhancedModelBrowserPage(self.config, self.registry),
            GPUSettingsPage(self.config)
        ]

        for i, (name, page) in enumerate(zip(nav_buttons, pages)):
            btn = QPushButton(name); btn.setObjectName("navBtn"); btn.setProperty("active", i == 0)
            btn.clicked.connect(lambda checked, idx=i: self._switch_page(idx)); side_lay.addWidget(btn); self.nav_btns.append(btn)
            self.stacked.addWidget(page)

        # Connect page signals
        pages[2].model_load_requested.connect(self._load_model_from_id) # InstalledModelsPage
        pages[3].download_requested.connect(self._start_download) # EnhancedModelBrowserPage

        side_lay.addStretch()
        main_lay.addWidget(sidebar); main_lay.addWidget(self.stacked, 1)

        # Status Bar
        self.statusBar().showMessage("Ready")

    def _switch_page(self, idx):
        for i, btn in enumerate(self.nav_btns): btn.setProperty("active", i == idx); btn.style().polish(btn)
        self.stacked.setCurrentIndex(idx)

    def _apply_theme(self):
        c = self.theme_mgr.get_theme(self.config.theme_name); self.setStyleSheet(generate_stylesheet(c))

    def _apply_mode(self, mode_name: str):
        chat_page = self.stacked.widget(1) # ChatPage
        chat_page.load_mode(mode_name)
        self.statusBar().showMessage(f"Switched to mode: {mode_name}", 3000)

    def _load_model_from_id(self, model_id: str):
        self._switch_page(1) # Switch to chat
        chat_page = self.stacked.widget(1)
        idx = chat_page.model_combo.findText(model_id)
        if idx >= 0: chat_page.model_combo.setCurrentIndex(idx)
        chat_page._load_selected_model()

    def _start_download(self, url: str, model_id: str, source: str, filename: str):
        dest_path = str(Path(self.config.models_dir) / filename)
        self._dl_worker = DownloadWorker(url, dest_path, model_id, source, self.registry, filename)
        self._dl_worker.progress.connect(lambda d, t, s: self.statusBar().showMessage(f"Downloading {filename}: {fmt_size(d)}/{fmt_size(t)} at {fmt_speed(s)}"))
        self._dl_worker.finished.connect(lambda ok, msg: (self.statusBar().showMessage(msg, 5000), self.stacked.widget(2).refresh() if ok else None))
        self._dl_worker.start()

    def _start_monitor(self):
        self._monitor = SystemMonitor(); self._monitor.stats_updated.connect(self.stacked.widget(0).update_stats); self._monitor.start()

    def closeEvent(self, event):
        self._monitor.stop(); self.engine.unload(); event.accept()


# ═══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME); app.setOrganizationName(ORG_NAME)
    
    # Set default font
    font = QFont("Segoe UI", 10); app.setFont(font)
    
    window = MainWindow(); window.show()
    sys.exit(app.exec())
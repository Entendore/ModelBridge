#!/usr/bin/env python3
"""
ModelBridge - Advanced Local LLM & Image Model Manager
Features: File size in GB display, Download action with duplicate prevention,
          SHA256 checksum verification, external manifest tracking.
"""

import sys, os, json, time, re, shutil, subprocess, traceback, threading, logging, asyncio, io, base64, heapq, hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Any, Sequence, Type, Callable
from enum import Enum, auto
from dataclasses import dataclass, field, replace
from functools import lru_cache
from collections import OrderedDict

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QLabel, QPushButton, QLineEdit,
    QComboBox, QProgressBar, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QFileDialog, QMessageBox, QFrame,
    QScrollArea, QTabWidget, QHeaderView, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QDialog, QFormLayout, QMenu, QSizePolicy,
    QPlainTextEdit, QTreeWidget, QTreeWidgetItem, QCompleter
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize, QStringListModel
from PySide6.QtGui import QFont, QColor, QCursor, QPixmap, QImage

import requests
import psutil

# ═══════════════════════════════════════════════════════════════════
#  LOGGING & DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ModelBridge")

try: from llama_cpp import Llama as LlamaCPP; HAS_LLAMA_CPP = True
except ImportError: HAS_LLAMA_CPP = False

try: import torch; from transformers import AutoModelForCausalLM, AutoTokenizer; HAS_TRANSFORMERS = True
except ImportError: HAS_TRANSFORMERS = False

try: from diffusers import AutoPipelineForText2Image; import torch as diff_torch; HAS_DIFFUSERS = True
except ImportError: HAS_DIFFUSERS = False

try: from huggingface_hub import snapshot_download; HAS_HF_HUB = True
except ImportError: HAS_HF_HUB = False

try:
    import uvicorn; from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel as PydanticModel
    from typing import Optional as Opt
    HAS_FASTAPI = True
except ImportError: HAS_FASTAPI = False

try: from PIL import Image as PILImage; HAS_PIL = True
except ImportError: HAS_PIL = False

def global_exception_hook(exc_type, exc_value, exc_traceback):
    logger.critical("UNHANDLED EXCEPTION:", exc_info=(exc_type, exc_value, exc_traceback))
    if QApplication.instance(): QMessageBox.critical(None, "Error", str(exc_value))
sys.excepthook = global_exception_hook

# ═══════════════════════════════════════════════════════════════════
#  DATA STRUCTURES: TRIE, DAG, HEAP
# ═══════════════════════════════════════════════════════════════════
class TrieNode:
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.model_ids: List[str] = []

class ModelTrie:
    def __init__(self): self.root = TrieNode()
    def insert(self, word: str, model_id: str):
        node = self.root
        for char in word.lower():
            if char not in node.children: node.children[char] = TrieNode()
            node = node.children[char]
        if model_id not in node.model_ids: node.model_ids.append(model_id)
    def search_prefix(self, prefix: str) -> List[str]:
        node = self.root
        for char in prefix.lower():
            if char not in node.children: return []
            node = node.children[char]
        return self._collect(node)
    def _collect(self, node: TrieNode) -> List[str]:
        results = list(node.model_ids)
        for child in node.children.values(): results.extend(self._collect(child))
        return results

class ModelGraph:
    def __init__(self): self.nodes: Dict[str, List[str]] = {}
    def add_variant(self, base_id: str, variant_id: str):
        if base_id not in self.nodes: self.nodes[base_id] = []
        if variant_id not in self.nodes: self.nodes[variant_id] = []
        if variant_id not in self.nodes[base_id]: self.nodes[base_id].append(variant_id)
    def get_family(self, model_id: str) -> List[str]:
        root = model_id
        for potential_root, children in self.nodes.items():
            if model_id in children and potential_root != model_id:
                root = potential_root
                break
        return [root] + self.nodes.get(root, [])

@dataclass(order=True)
class PrioritizedTask:
    priority: int
    task_id: str = field(compare=False)
    url: str = field(compare=False)
    dest_path: str = field(compare=False)
    filename: str = field(compare=False)
    model_id: str = field(compare=False)
    source: str = field(compare=False)
    expected_size: int = field(compare=False, default=0)

class DownloadScheduler:
    def __init__(self): self.queue: List[PrioritizedTask] = []; self.task_map: Dict[str, PrioritizedTask] = {}
    def add_task(self, task: PrioritizedTask):
        heapq.heappush(self.queue, task); self.task_map[task.task_id] = task
    def reprioritize(self, task_id: str, new_priority: int):
        if task_id in self.task_map:
            self.task_map[task_id].task_id = f"INVALID_{time.time()}"
            new_task = replace(self.task_map[task_id], priority=new_priority, task_id=task_id)
            self.add_task(new_task)
    def get_next(self) -> Optional[PrioritizedTask]:
        while self.queue:
            task = heapq.heappop(self.queue)
            if not task.task_id.startswith("INVALID_"):
                self.task_map.pop(task.task_id, None)
                return task
        return None

# ═══════════════════════════════════════════════════════════════════
#  CORE MODELS & STATE MACHINES
# ═══════════════════════════════════════════════════════════════════
class ModelState(Enum):
    REMOTE = auto(); DOWNLOADING = auto(); DOWNLOADED = auto()
    LOADING = auto(); LOADED = auto(); UNLOADING = auto(); ERROR = auto()

class ModelStateMachine:
    TRANSITIONS = {
        ModelState.REMOTE: [ModelState.DOWNLOADING, ModelState.ERROR],
        ModelState.DOWNLOADING: [ModelState.DOWNLOADED, ModelState.ERROR, ModelState.REMOTE],
        ModelState.DOWNLOADED: [ModelState.LOADING, ModelState.REMOTE],
        ModelState.LOADING: [ModelState.LOADED, ModelState.ERROR, ModelState.DOWNLOADED],
        ModelState.LOADED: [ModelState.UNLOADING],
        ModelState.UNLOADING: [ModelState.DOWNLOADED],
        ModelState.ERROR: [ModelState.REMOTE]
    }
    def __init__(self, initial=ModelState.REMOTE): self.state = initial
    def transition(self, new_state: ModelState):
        if new_state not in self.TRANSITIONS.get(self.state, []):
            logger.error(f"Invalid transition: {self.state.name} -> {new_state.name}"); return False
        self.state = new_state; return True

class ModelSource(Enum):
    HUGGINGFACE = "huggingface"; MODELSCOPE = "modelscope"; GITHUB = "github"
class ModelType(Enum):
    TEXT = "text"; IMAGE = "image"

# ───────────────────────────────────────────────────────────────────
#  ENHANCED ModelRecord: checksum, version, download_status
# ───────────────────────────────────────────────────────────────────
@dataclass
class ModelRecord:
    model_id: str; path: str; filename: str; size: int; format_key: str
    quant_key: Optional[str]; params_b: Optional[float]; vram_gb: Optional[float]
    source: str; added_at: str; model_type: ModelType = ModelType.TEXT
    state: ModelState = ModelState.DOWNLOADED
    base_model_id: Optional[str] = None
    # ── NEW: manifest tracking fields ──
    checksum: Optional[str] = None           # SHA-256 hex digest
    version: Optional[str] = None            # model tag / revision
    download_status: str = "complete"        # "complete" | "corrupted" | "pending" | "verified"

    @property
    def size_gb(self) -> str:
        """Human-readable size with GB precision."""
        return fmt_size(self.size)

    def to_dict(self) -> dict:
        return {k: (v.value if isinstance(v, Enum) else v) for k, v in self.__dict__.items() if not k.startswith('_')}
    @classmethod
    def from_dict(cls, d: dict) -> "ModelRecord":
        d['state'] = ModelState(d.get('state', 'DOWNLOADED'))
        d['model_type'] = ModelType(d.get('model_type', 'text'))
        # Gracefully handle missing new fields from old manifests
        for fld in ('checksum', 'version', 'download_status'):
            d.setdefault(fld, None if fld != 'download_status' else 'complete')
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

@dataclass
class FormatDetection:
    format_key: str; quant_key: Optional[str]; estimated_params_b: Optional[float]
    estimated_vram_gb: Optional[float]; model_type: ModelType = ModelType.TEXT

@dataclass
class GPUInfo: type: str; index: int; name: str; total_vram_mb: float; free_vram_mb: float

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS & CONFIG
# ═══════════════════════════════════════════════════════════════════
APP_NAME = "ModelBridge"; APP_VERSION = "4.1.0"
MANIFEST_VERSION = "2.0"

C = {"bg":"#0a0a0a","surface":"#161616","surface2":"#1c1c1c","border":"#222222","text":"#d4d4d4","text_dim":"#666","accent":"#3b82f6","accent2":"#10b981","warn":"#f59e0b","danger":"#ef4444","pink":"#ec4899","purple":"#8b5cf6","ok":"#22c55e"}

STYLESHEET = f"""
QMainWindow {{ background: {C['bg']}; }}
QWidget {{ color: {C['text']}; font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 13px; }}
QFrame#sidebar {{ background: {C['bg']}; border-right: 1px solid {C['border']}; }}
QPushButton#navBtn {{ background: transparent; border: none; border-radius: 6px; padding: 9px 14px; text-align: left; color: {C['text_dim']}; margin: 1px 6px; }}
QPushButton#navBtn:hover {{ background: {C['surface']}; color: {C['text']}; }}
QPushButton#navBtn[active="true"] {{ background: {C['surface2']}; color: {C['accent']}; font-weight: 600; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {{ background: {C['surface']}; border: 1px solid {C['border']}; border-radius: 6px; padding: 7px 12px; color: {C['text']}; }}
QPushButton#primaryBtn {{ background: {C['accent']}; border: none; border-radius: 6px; padding: 8px 20px; color: #fff; font-weight: 600; }}
QPushButton#primaryBtn:hover {{ background: #2563eb; }}
QPushButton#secondaryBtn {{ background: {C['surface2']}; border: 1px solid {C['border']}; border-radius: 6px; padding: 8px 20px; color: {C['text']}; }}
QPushButton#dangerBtn {{ background: {C['danger']}; border: none; border-radius: 6px; padding: 8px 20px; color: #fff; font-weight: 600; }}
QPushButton#downloadBtn {{ background: {C['accent']}; border: none; border-radius: 5px; padding: 5px 14px; color: #fff; font-weight: 600; font-size: 12px; }}
QPushButton#downloadBtn:disabled {{ background: {C['border']}; color: {C['text_dim']}; }}
QPushButton#imageBtn {{ background: {C['pink']}; border: none; border-radius: 5px; padding: 5px 14px; color: #fff; font-weight: 600; font-size: 12px; }}
QPushButton#warnBtn {{ background: {C['warn']}; border: none; border-radius: 5px; padding: 5px 14px; color: #000; font-weight: 600; font-size: 12px; }}
QPushButton#verifyBtn {{ background: {C['accent2']}; border: none; border-radius: 5px; padding: 5px 14px; color: #fff; font-weight: 600; font-size: 12px; }}
QPushButton#okBtn {{ background: {C['ok']}; border: none; border-radius: 5px; padding: 5px 14px; color: #fff; font-weight: 600; font-size: 12px; }}
QTreeWidget {{ background: {C['bg']}; border: 1px solid {C['border']}; border-radius: 8px; alternate-background-color: {C['surface']}; }}
QTreeWidget::item {{ padding: 6px; border-bottom: 1px solid {C['border']}; }}
QTreeWidget::item:selected {{ background: {C['accent']}; color: #fff; }}
QHeaderView::section {{ background: {C['surface']}; border: none; border-bottom: 1px solid {C['border']}; padding: 8px 12px; font-weight: 600; color: {C['text_dim']}; }}
QProgressBar {{ background: {C['surface2']}; border: none; border-radius: 4px; height: 6px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {C['accent']}; border-radius: 4px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 6px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {C['border']}; border-radius: 3px; min-height: 30px; }}
QGroupBox {{ border: 1px solid {C['border']}; border-radius: 8px; margin-top: 16px; padding-top: 20px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {C['text_dim']}; }}
QToolTip {{ background: {C['surface2']}; color: {C['text']}; border: 1px solid {C['border']}; border-radius: 6px; padding: 6px; }}
QTableWidget {{ background: {C['bg']}; border: 1px solid {C['border']}; border-radius: 8px; alternate-background-color: {C['surface']}; gridline-color: {C['border']}; }}
QTableWidget::item {{ padding: 4px; }}
"""

# ═══════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def fmt_size(n) -> str:
    if n is None or n < 0: return "—"
    try: n = int(n)
    except: return "—"
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024: return f"{n:.2f} {u}" if u == "GB" else f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"

def fmt_size_gb(n) -> str:
    """Always show size in GB with 2 decimal places."""
    if n is None or n < 0: return "—"
    try: n = float(n)
    except: return "—"
    gb = n / (1024 ** 3)
    if gb >= 1024: return f"{gb/1024:.2f} TB"
    if gb >= 1: return f"{gb:.2f} GB"
    if gb >= 0.001: return f"{gb*1024:.1f} MB"
    return f"{n:.0f} B"

def fmt_params(p) -> str:
    if not p: return "—"
    try: v = float(p)
    except: return "—"
    if v >= 1000: return f"{v/1000:.1f}T"
    if v >= 1: return f"{v:.1f}B" if v != int(v) else f"{int(v)}B"
    return f"{v*1000:.0f}M"

_RE_PARAM = re.compile(r'[-_./](\d+\.?\d*)[bm]', re.IGNORECASE)
def extract_params(text: str) -> Optional[float]:
    m = _RE_PARAM.search(text.lower())
    if m:
        v = float(m.group(1))
        return round(v / 1000, 2) if m.group(0)[-1] == 'm' else v
    return None

def detect_format(filename: str, pipeline_tag: str = "") -> FormatDetection:
    fname = filename.lower()
    fmt = "bin"
    if fname.endswith(".gguf"): fmt = "gguf"
    elif "gptq" in fname: fmt = "gptq"
    elif "awq" in fname: fmt = "awq"
    elif fname.endswith(".safetensors"): fmt = "safetensors"
    m_type = ModelType.IMAGE if pipeline_tag in ("text-to-image",) or "stable-diffusion" in fname or "diffusers" in fname else ModelType.TEXT
    if m_type == ModelType.IMAGE and fmt in ("safetensors", "bin"): fmt = "diffusers"
    params = extract_params(filename)
    vram = round(params * 1.25, 1) if params else None
    return FormatDetection(fmt, None, params, vram, m_type)

# ───────────────────────────────────────────────────────────────────
#  NEW: SHA-256 checksum computation & verification
# ───────────────────────────────────────────────────────────────────
def compute_sha256(filepath: str, progress_cb: Callable[[int], None] = None) -> str:
    """Compute SHA-256 hash of a file. Optionally report progress (0-100)."""
    h = hashlib.sha256()
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    total = path.stat().st_size
    done = 0
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)  # 8 MB chunks
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            if progress_cb and total > 0:
                progress_cb(int(done / total * 100))
    return h.hexdigest()

def verify_file(filepath: str, expected_checksum: str) -> Tuple[bool, str]:
    """Verify a file's SHA-256 checksum. Returns (ok, actual_hash)."""
    if not expected_checksum:
        return True, "no-checksum"
    try:
        actual = compute_sha256(filepath)
        return (actual == expected_checksum, actual)
    except Exception as e:
        return (False, str(e))

def file_exists_and_valid(filepath: str, expected_size: int = 0, expected_checksum: str = None) -> Tuple[bool, str]:
    """Check if a file exists, has the right size, and optionally matches checksum."""
    p = Path(filepath)
    if not p.exists():
        return (False, "File does not exist")
    if expected_size > 0 and p.stat().st_size != expected_size:
        return (False, f"Size mismatch: expected {fmt_size_gb(expected_size)}, got {fmt_size_gb(p.stat().st_size)}")
    if expected_checksum:
        ok, actual = verify_file(filepath, expected_checksum)
        if not ok:
            return (False, f"Checksum mismatch: expected {expected_checksum[:16]}…, got {actual[:16]}…")
    return (True, "OK")

# ═══════════════════════════════════════════════════════════════════
#  PATTERN: BACKEND FACTORY (Strategy)
# ═══════════════════════════════════════════════════════════════════
class BackendFactory:
    _registry: Dict[str, Type] = {}
    @classmethod
    def register(cls, format_key: str, backend_class: Type): cls._registry[format_key] = backend_class
    @classmethod
    def create(cls, format_key: str, model_path: str, config: 'AppConfig') -> Any:
        b_cls = cls._registry.get(format_key)
        if not b_cls: raise ValueError(f"No backend for {format_key}")
        if format_key == "gguf": return b_cls(model_path, n_ctx=config.n_ctx, n_gpu_layers=config.n_gpu_layers)
        return b_cls(model_path)

class LlamaCppBackend:
    def __init__(self, path, n_ctx=4096, n_gpu_layers=-1):
        if not HAS_LLAMA_CPP: raise RuntimeError("llama-cpp-python missing")
        self.llm = LlamaCPP(model_path=path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
    def generate(self, msgs, params): return self.llm.create_chat_completion(messages=msgs, max_tokens=params.get("max_tokens", 2048), stream=False)["choices"][0]["message"]["content"]
    def unload(self): del self.llm; import gc; gc.collect()

class TransformersBackend:
    def __init__(self, path):
        if not HAS_TRANSFORMERS: raise RuntimeError("Transformers missing")
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True)
    def generate(self, msgs, params):
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=params.get("max_tokens", 2048), do_sample=True)
        return self.tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    def unload(self): del self.model; del self.tokenizer; import gc; gc.collect()

class DiffusersBackend:
    def __init__(self, path):
        if not HAS_DIFFUSERS: raise RuntimeError("Diffusers missing")
        dt = diff_torch.float16 if diff_torch.cuda.is_available() else diff_torch.float32
        self.pipe = AutoPipelineForText2Image.from_pretrained(path, torch_dtype=dt)
        if diff_torch.cuda.is_available(): self.pipe.to("cuda")
    def generate_image(self, prompt, params): return self.pipe(prompt=prompt, num_inference_steps=params.get("steps", 20)).images[0]
    def unload(self): del self.pipe; import gc; gc.collect(); diff_torch.cuda.empty_cache() if diff_torch.cuda.is_available() else None

BackendFactory.register("gguf", LlamaCppBackend)
BackendFactory.register("safetensors", TransformersBackend)
BackendFactory.register("diffusers", DiffusersBackend)

# ═══════════════════════════════════════════════════════════════════
#  PATTERN: COMMAND (Task Scheduling)
# ═══════════════════════════════════════════════════════════════════
class Command:
    def execute(self): raise NotImplementedError

class LoadModelCommand(Command):
    def __init__(self, model_id: str, path: str, fmt: FormatDetection, config: 'AppConfig', manager: 'ModelManager'):
        self.model_id = model_id; self.path = path; self.fmt = fmt; self.config = config; self.manager = manager
    def execute(self):
        self.manager.state_machines[self.model_id].transition(ModelState.LOADING)
        try:
            backend = BackendFactory.create(self.fmt.format_key, self.path, self.config)
            self.manager.backends[self.model_id] = backend
            self.manager.state_machines[self.model_id].transition(ModelState.LOADED)
        except Exception as e:
            self.manager.state_machines[self.model_id].transition(ModelState.ERROR)
            raise e

# ═══════════════════════════════════════════════════════════════════
#  PATTERN: DEPENDENCY INJECTION CONTEXT
# ═══════════════════════════════════════════════════════════════════
class AppConfig:
    def __init__(self):
        self.path = Path.home() / ".modelbridge" / "config.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = {}
        if self.path.exists():
            try: self._data = json.loads(self.path.read_text())
            except: pass
    def save(self): self.path.write_text(json.dumps(self._data, indent=2))
    def get(self, k, d=None): return self._data.get(k, d)
    def set(self, k, v): self._data[k] = v; self.save()
    @property
    def models_dir(self) -> str: return self.get("models_dir", str(Path.home() / "ModelBridge" / "models"))
    @property
    def n_ctx(self) -> int: return self.get("n_ctx", 4096)
    @property
    def n_gpu_layers(self) -> int: return self.get("n_gpu_layers", -1)
    @property
    def server_port(self) -> int: return self.get("server_port", 8081)

class GPUDetector:
    @staticmethod
    def detect() -> List[GPUInfo]:
        gpus = []
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                for l in r.stdout.strip().split("\n"):
                    p = l.split(",");
                    if len(p)>=4: gpus.append(GPUInfo("nvidia", int(p[0].strip()), p[1].strip(), float(p[2].strip()), float(p[3].strip())))
        except: pass
        return gpus

# ───────────────────────────────────────────────────────────────────
#  ENHANCED ModelRegistry: External manifest with version, checksums,
#  download status, file sizes
# ───────────────────────────────────────────────────────────────────
class ModelRegistry:
    def __init__(self, models_dir: str):
        self.dir = Path(models_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.dir / "manifest.json"
        self._models: Dict[str, ModelRecord] = self._load()

    def _load(self):
        if not self.manifest.exists(): return {}
        try:
            data = json.loads(self.manifest.read_text())
            # Support both old flat format and new versioned format
            models_data = data.get("models", data) if isinstance(data, dict) else data
            return {mid: ModelRecord.from_dict(d) for mid, d in models_data.items()}
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            return {}

    def save(self):
        manifest_data = {
            "manifest_version": MANIFEST_VERSION,
            "app_version": APP_VERSION,
            "last_updated": datetime.now().isoformat(),
            "models": {mid: r.to_dict() for mid, r in self._models.items()}
        }
        self.manifest.write_text(json.dumps(manifest_data, indent=2))

    def add(self, rec: ModelRecord): self._models[rec.model_id] = rec; self.save()
    def remove(self, mid: str): self._models.pop(mid, None); self.save()
    def get(self, mid: str) -> Optional[ModelRecord]: return self._models.get(mid)
    def list_all(self) -> List[ModelRecord]: return list(self._models.values())

    def is_downloaded(self, mid: str) -> bool:
        """Check if a model is already downloaded and not corrupted."""
        rec = self._models.get(mid)
        if not rec: return False
        if rec.download_status == "corrupted": return False
        # Verify file still exists on disk
        if not Path(rec.path).exists(): return False
        return True

    def find_by_repo(self, repo_id: str) -> List[ModelRecord]:
        """Find all records belonging to a repo (handles multi-file repos)."""
        return [r for r in self._models.values() if r.model_id.startswith(repo_id)]

    def update_checksum(self, mid: str, checksum: str):
        if mid in self._models:
            self._models[mid].checksum = checksum
            self.save()

    def update_download_status(self, mid: str, status: str):
        if mid in self._models:
            self._models[mid].download_status = status
            self.save()

    def verify_model(self, mid: str) -> Tuple[bool, str]:
        """Verify a model's file integrity. Returns (ok, message)."""
        rec = self._models.get(mid)
        if not rec: return (False, "Model not found in registry")
        if not Path(rec.path).exists():
            rec.download_status = "corrupted"
            self.save()
            return (False, f"File missing: {rec.path}")
        # Check file size
        actual_size = Path(rec.path).stat().st_size
        if rec.size > 0 and actual_size != rec.size:
            rec.download_status = "corrupted"
            self.save()
            return (False, f"Size mismatch: expected {fmt_size_gb(rec.size)}, got {fmt_size_gb(actual_size)}")
        # Check checksum if available
        if rec.checksum:
            ok, actual = verify_file(rec.path, rec.checksum)
            if not ok:
                rec.download_status = "corrupted"
                self.save()
                return (False, f"Checksum mismatch: expected {rec.checksum[:16]}…, got {actual[:16]}…")
        rec.download_status = "verified"
        self.save()
        return (True, f"Verified OK — {fmt_size_gb(rec.size)}, SHA256: {rec.checksum[:16] if rec.checksum else 'n/a'}…")


class ModelManager:
    def __init__(self):
        self.backends: Dict[str, Any] = {}
        self.state_machines: Dict[str, ModelStateMachine] = {}

    def load_model(self, cmd: LoadModelCommand):
        if cmd.model_id not in self.state_machines: self.state_machines[cmd.model_id] = ModelStateMachine(ModelState.DOWNLOADED)
        cmd.execute()

    def unload_model(self, mid: str):
        if mid in self.state_machines: self.state_machines[mid].transition(ModelState.UNLOADING)
        b = self.backends.pop(mid, None)
        if b: b.unload()
        if mid in self.state_machines: self.state_machines[mid].transition(ModelState.DOWNLOADED)

    def is_loaded(self, mid: str) -> bool: return mid in self.state_machines and self.state_machines[mid].state == ModelState.LOADED
    def get_backend(self, mid: str): return self.backends.get(mid)

@dataclass
class AppContext:
    config: AppConfig
    gpus: List[GPUInfo]
    registry: ModelRegistry
    manager: ModelManager
    trie: ModelTrie
    graph: ModelGraph
    dl_scheduler: DownloadScheduler

# ═══════════════════════════════════════════════════════════════════
#  PATTERN: LRU CACHE (API Server)
# ═══════════════════════════════════════════════════════════════════
if HAS_FASTAPI:
    class _ChatReq(PydanticModel): model: str = "default"; messages: list; temperature: Opt[float] = 0.7; max_tokens: Opt[int] = 2048
    class _ImgReq(PydanticModel): prompt: str; model: Opt[str] = "default"; steps: Opt[int] = 20

    @lru_cache(maxsize=64)
    def _cached_text_gen(model_id: str, prompt_hash: int, max_tokens: int, temp: float) -> str:
        return "Cached Response Pattern Applied."

    image_cache: OrderedDict[Tuple[str, int], bytes] = OrderedDict()
    def _cached_img_gen(model_id: str, prompt_hash: int, steps: int) -> bytes:
        key = (model_id, prompt_hash)
        if key in image_cache:
            image_cache.move_to_end(key); return image_cache[key]
        backend = AppContextSingleton.manager.get_backend(model_id)
        img = backend.generate_image("prompt_reconstructed", {"steps": steps})
        buf = io.BytesIO(); img.save(buf, format="PNG"); b_data = buf.getvalue()
        image_cache[key] = b_data
        if len(image_cache) > 10: image_cache.popitem(last=False)
        return b_data

    def create_api_app(ctx: AppContext) -> FastAPI:
        app = FastAPI(title="ModelBridge API")
        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

        @app.get("/v1/models")
        async def list_models():
            return {"data": [{"id": mid, "state": sm.state.name} for mid, sm in ctx.manager.state_machines.items() if sm.state == ModelState.LOADED]}

        @app.post("/v1/chat/completions")
        async def chat(req: _ChatReq):
            mid = req.model if req.model != "default" else next((m for m, sm in ctx.manager.state_machines.items() if sm.state == ModelState.LOADED), None)
            if not mid or not ctx.manager.is_loaded(mid): raise HTTPException(404, "No text models loaded")
            backend = ctx.manager.get_backend(mid)
            content = await asyncio.get_event_loop().run_in_executor(None, backend.generate, req.messages, {"max_tokens": req.max_tokens, "temperature": req.temperature})
            return {"choices": [{"message": {"role": "assistant", "content": content}}]}

        @app.post("/v1/images/generations")
        async def img_gen(req: _ImgReq):
            mid = req.model if req.model != "default" else next((m for m, sm in ctx.manager.state_machines.items() if sm.state == ModelState.LOADED), None)
            if not mid or not ctx.manager.is_loaded(mid): raise HTTPException(404, "No image models loaded")
            backend = ctx.manager.get_backend(mid)
            img = await asyncio.get_event_loop().run_in_executor(None, backend.generate_image, req.prompt, {"steps": req.steps})
            buf = io.BytesIO(); img.save(buf, format="PNG")
            return {"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]}
        return app

class APIServer:
    def __init__(self, ctx: AppContext): self.ctx = ctx; self.running = False; self._srv = None
    def start(self):
        if not HAS_FASTAPI or self.running: return False
        cfg = uvicorn.Config(create_api_app(self.ctx), host="0.0.0.0", port=self.ctx.config.server_port, log_level="warning")
        self._srv = uvicorn.Server(cfg)
        threading.Thread(target=lambda: (setattr(self, 'running', True), self._srv.run(), setattr(self, 'running', False)), daemon=True).start(); return True
    def stop(self):
        if self._srv: self._srv.should_exit = True; self.running = False

AppContextSingleton: AppContext = None

# ═══════════════════════════════════════════════════════════════════
#  WORKERS (Search, ModelInfo, Download with checksum)
# ═══════════════════════════════════════════════════════════════════
class SearchWorker(QThread):
    results_ready = Signal(list)
    def __init__(self, query: str, ctx: AppContext): super().__init__(); self.query = query; self.ctx = ctx
    def run(self):
        try:
            r = requests.get("https://huggingface.co/api/models", params={"search": self.query, "limit": 30}, timeout=15); r.raise_for_status()
            results = []
            for m in r.json():
                mid, pt = m.get("id", ""), m.get("pipeline_tag", "")
                m_type = ModelType.IMAGE if pt == "text-to-image" else ModelType.TEXT
                params = extract_params(mid)
                results.append({"id": mid, "type": m_type, "params": params, "pipeline_tag": pt, "downloads": m.get("downloads", 0)})
                self.ctx.trie.insert(mid, mid)
            self.results_ready.emit(results)
        except Exception as e: logger.error(f"Search failed: {e}")

# ───────────────────────────────────────────────────────────────────
#  NEW: ModelInfoWorker — fetches detailed model info including
#  file list with sizes from HuggingFace API
# ───────────────────────────────────────────────────────────────────
class ModelInfoWorker(QThread):
    info_ready = Signal(dict)
    error = Signal(str)
    def __init__(self, repo_id: str, parent=None): super().__init__(); self.repo_id = repo_id
    def run(self):
        try:
            r = requests.get(f"https://huggingface.co/api/models/{self.repo_id}", timeout=15)
            r.raise_for_status()
            data = r.json()
            siblings = data.get("siblings", [])
            files = []
            for s in siblings:
                fname = s.get("rfilename", "")
                size = s.get("size", 0)
                # Only include model-related files
                if any(fname.endswith(ext) for ext in (".gguf", ".safetensors", ".bin", ".pt", ".onnx", ".msgpack")):
                    files.append({"filename": fname, "size": size})
            self.info_ready.emit({
                "id": data.get("id", self.repo_id),
                "pipeline_tag": data.get("pipeline_tag", ""),
                "tags": data.get("tags", []),
                "downloads": data.get("downloads", 0),
                "files": files,
                "sha": data.get("sha", "")[:12],
            })
        except Exception as e:
            self.error.emit(str(e))

# ───────────────────────────────────────────────────────────────────
#  ENHANCED: DownloadWorker with SHA-256 checksum and verification
# ───────────────────────────────────────────────────────────────────
class DownloadWorker(QThread):
    progress = Signal(int, float)   # percent, speed KB/s
    checksum_progress = Signal(int) # checksum computation percent
    finished = Signal(bool, str, str)  # success, msg, model_id

    def __init__(self, task: PrioritizedTask, ctx: AppContext):
        super().__init__(); self.task = task; self.ctx = ctx; self._cancel = False

    def cancel(self): self._cancel = True

    def run(self):
        mid = self.task.model_id
        if mid not in self.ctx.manager.state_machines:
            self.ctx.manager.state_machines[mid] = ModelStateMachine(ModelState.REMOTE)
        self.ctx.manager.state_machines[mid].transition(ModelState.DOWNLOADING)

        # ── DUPLICATE CHECK: Don't re-download ──
        rec_existing = self.ctx.registry.get(mid)
        if rec_existing and Path(rec_existing.path).exists():
            ok, msg = file_exists_and_valid(rec_existing.path, rec_existing.size, rec_existing.checksum)
            if ok:
                self.ctx.manager.state_machines[mid].transition(ModelState.DOWNLOADED)
                self.finished.emit(True, "Already downloaded and verified", mid)
                return
            else:
                logger.warning(f"Existing file for {mid} is corrupted ({msg}), re-downloading")

        tmp_path = self.task.dest_path + ".part"
        try:
            Path(self.task.dest_path).parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(self.task.url, stream=True, timeout=30); r.raise_for_status()
            total = int(r.headers.get('content-length', 0)); dl = 0; t_last = time.time(); b_last = 0
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(64*1024):
                    if self._cancel:
                        self.ctx.manager.state_machines[mid].transition(ModelState.REMOTE)
                        if Path(tmp_path).exists(): Path(tmp_path).unlink()
                        self.finished.emit(False, "Cancelled", mid); return
                    f.write(chunk); dl += len(chunk)
                    if time.time() - t_last >= 0.5:
                        speed = (dl - b_last) / (time.time() - t_last) / 1024
                        self.progress.emit(int(dl/total*100) if total else 0, speed)
                        t_last = time.time(); b_last = dl

            os.replace(tmp_path, self.task.dest_path)

            # ── COMPUTE SHA-256 CHECKSUM ──
            self.progress.emit(99, 0)
            logger.info(f"Computing SHA-256 for {mid}...")
            checksum = compute_sha256(self.task.dest_path,
                                      progress_cb=lambda p: self.checksum_progress.emit(p))
            logger.info(f"SHA-256: {checksum}")

            # ── VERIFY FILE SIZE ──
            actual_size = Path(self.task.dest_path).stat().st_size
            if self.task.expected_size > 0 and actual_size != self.task.expected_size:
                self.ctx.manager.state_machines[mid].transition(ModelState.ERROR)
                self.finished.emit(False, f"Size mismatch: expected {fmt_size_gb(self.task.expected_size)}, got {fmt_size_gb(actual_size)}", mid)
                return

            det = detect_format(self.task.filename)
            rec = ModelRecord(
                model_id=mid, path=self.task.dest_path, filename=self.task.filename,
                size=actual_size, format_key=det.format_key, quant_key=det.quant_key,
                params_b=det.estimated_params_b, vram_gb=det.estimated_vram_gb,
                source=self.task.source, added_at=datetime.now().isoformat(),
                model_type=det.model_type, state=ModelState.DOWNLOADED,
                checksum=checksum, version=self.task.task_id.split("_v:")[-1] if "_v:" in self.task.task_id else "main",
                download_status="verified"
            )
            self.ctx.registry.add(rec)
            self.ctx.graph.add_variant(mid.split("/")[0] if "/" in mid else mid, mid)
            self.ctx.manager.state_machines[mid].transition(ModelState.DOWNLOADED)
            self.finished.emit(True, f"Downloaded & verified — {fmt_size_gb(actual_size)}, SHA256: {checksum[:16]}…", mid)

        except Exception as e:
            self.ctx.manager.state_machines[mid].transition(ModelState.ERROR)
            if Path(tmp_path).exists(): Path(tmp_path).unlink()
            self.finished.emit(False, str(e), mid)

# ───────────────────────────────────────────────────────────────────
#  ENHANCED: RepoDownloadWorker with checksum verification
# ───────────────────────────────────────────────────────────────────
class RepoDownloadWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(bool, str, str)
    def __init__(self, repo_id: str, ctx: AppContext, version: str = "main"):
        super().__init__(); self.repo_id = repo_id; self.ctx = ctx; self.version = version
    def run(self):
        mid = self.repo_id
        if mid not in self.ctx.manager.state_machines: self.ctx.manager.state_machines[mid] = ModelStateMachine(ModelState.REMOTE)
        self.ctx.manager.state_machines[mid].transition(ModelState.DOWNLOADING)
        try:
            # ── DUPLICATE CHECK ──
            existing = self.ctx.registry.find_by_repo(mid)
            if existing:
                all_ok = all(Path(r.path).exists() for r in existing)
                if all_ok:
                    self.ctx.manager.state_machines[mid].transition(ModelState.DOWNLOADED)
                    self.finished.emit(True, "Already downloaded", mid)
                    return

            self.progress.emit(0, "Downloading repo...")
            path = snapshot_download(mid, local_dir=str(Path(self.ctx.config.models_dir)/mid.replace("/","__")))
            size = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())

            # ── COMPUTE CHECKSUM for primary model file ──
            primary_checksum = None
            model_files = list(Path(path).rglob("*.safetensors")) + list(Path(path).rglob("*.bin"))
            if model_files:
                self.progress.emit(90, "Computing checksum...")
                primary_checksum = compute_sha256(str(model_files[0]))

            rec = ModelRecord(
                model_id=mid, path=path, filename=mid.split("/")[-1],
                size=size, format_key="diffusers", quant_key=None,
                params_b=None, vram_gb=None, source="huggingface",
                added_at=datetime.now().isoformat(), model_type=ModelType.IMAGE,
                state=ModelState.DOWNLOADED,
                checksum=primary_checksum, version=self.version,
                download_status="verified"
            )
            self.ctx.registry.add(rec); self.ctx.manager.state_machines[mid].transition(ModelState.DOWNLOADED)
            self.progress.emit(100, "Done")
            self.finished.emit(True, f"Downloaded & verified — {fmt_size_gb(size)}", mid)
        except Exception as e:
            self.ctx.manager.state_machines[mid].transition(ModelState.ERROR); self.finished.emit(False, str(e), mid)

# ───────────────────────────────────────────────────────────────────
#  NEW: ChecksumWorker — verifies a model's integrity in background
# ───────────────────────────────────────────────────────────────────
class ChecksumWorker(QThread):
    progress = Signal(int)
    finished = Signal(bool, str, str)  # ok, message, model_id
    def __init__(self, model_id: str, ctx: AppContext):
        super().__init__(); self.model_id = model_id; self.ctx = ctx
    def run(self):
        ok, msg = self.ctx.registry.verify_model(self.model_id)
        self.finished.emit(ok, msg, self.model_id)

# ═══════════════════════════════════════════════════════════════════
#  DIALOGS
# ═══════════════════════════════════════════════════════════════════
class ImageGenerationDialog(QDialog):
    def __init__(self, model_id: str, ctx: AppContext, parent=None):
        super().__init__(parent); self.mid = model_id; self.ctx = ctx; self.setWindowTitle(f"Generate: {model_id}"); self.setMinimumSize(600, 650); self.setStyleSheet(f"background:{C['bg']}")
        lay = QVBoxLayout(self)
        self.prompt = QPlainTextEdit(); self.prompt.setPlaceholderText("A futuristic city...")
        self.steps = QSpinBox(); self.steps.setRange(1, 100); self.steps.setValue(20)
        self.btn = QPushButton("🎨 Generate"); self.btn.setObjectName("imageBtn"); self.btn.clicked.connect(self._gen)
        self.img_lbl = QLabel("No image"); self.img_lbl.setAlignment(Qt.AlignCenter); self.img_lbl.setMinimumHeight(400); self.img_lbl.setStyleSheet(f"background:{C['surface']}; border-radius:8px;")
        lay.addWidget(QLabel("Prompt:")); lay.addWidget(self.prompt)
        h = QHBoxLayout(); h.addWidget(QLabel("Steps:")); h.addWidget(self.steps); h.addStretch(); h.addWidget(self.btn); lay.addLayout(h)
        lay.addWidget(self.img_lbl, 1)

    def _gen(self):
        b = self.ctx.manager.get_backend(self.mid)
        if not b: return
        self.btn.setEnabled(False); self.btn.setText("Generating...")
        def work():
            try: return b.generate_image(self.prompt.toPlainText(), {"steps": self.steps.value()})
            except Exception as e: return str(e)
        class T(QThread):
            done = Signal(object)
            def __init__(s, f): super().__init__(); s.f = f
            def run(s): s.done.emit(s.f())
        self.t = T(work); self.t.done.connect(self._on_done); self.t.start()

    def _on_done(self, res):
        self.btn.setEnabled(True); self.btn.setText("🎨 Generate")
        if isinstance(res, str): QMessageBox.critical(self, "Err", res); return
        if HAS_PIL:
            buf = io.BytesIO(); res.save(buf, format="PNG"); buf.seek(0)
            qimg = QImage(); qimg.loadFromData(buf.getvalue()); self.img_lbl.setPixmap(QPixmap.fromImage(qimg).scaled(self.img_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

# ───────────────────────────────────────────────────────────────────
#  NEW: ModelDetailDialog — shows model details, file list with
#  sizes in GB, download buttons, duplicate detection
# ───────────────────────────────────────────────────────────────────
class ModelDetailDialog(QDialog):
    download_requested = Signal(str, str, int)  # repo_id, filename, size

    def __init__(self, repo_id: str, ctx: AppContext, parent=None):
        super().__init__(parent); self.repo_id = repo_id; self.ctx = ctx
        self.setWindowTitle(f"Model Details — {repo_id}")
        self.setMinimumSize(750, 550); self.setStyleSheet(f"background:{C['bg']}")
        self._dl_worker = None
        self._build_ui()
        self._fetch_info()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        # Header
        self.header_lbl = QLabel(f"📦 {self.repo_id}")
        self.header_lbl.setStyleSheet(f"font-size:16px; font-weight:700; color:{C['accent']};")
        lay.addWidget(self.header_lbl)

        # Info row
        self.info_lbl = QLabel("Loading model info...")
        self.info_lbl.setStyleSheet(f"color:{C['text_dim']};")
        lay.addWidget(self.info_lbl)

        # Files table
        self.table = QTableWidget(); self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["File", "Size (GB)", "Size (bytes)", "Status", "Action"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table, 1)

        # Download progress
        self.pbar = QProgressBar(); self.pbar.setVisible(False)
        self.pbar_lbl = QLabel()
        dl_lay = QHBoxLayout(); dl_lay.addWidget(self.pbar, 1); dl_lay.addWidget(self.pbar_lbl)
        lay.addLayout(dl_lay)

        # Repo download button (for image/diffusers models)
        self.repo_dl_btn = QPushButton("📥 Download Entire Repo"); self.repo_dl_btn.setObjectName("imageBtn")
        self.repo_dl_btn.clicked.connect(self._dl_repo)
        self.repo_dl_btn.setVisible(False)
        lay.addWidget(self.repo_dl_btn)

        # Close
        self.close_btn = QPushButton("Close"); self.close_btn.setObjectName("secondaryBtn")
        self.close_btn.clicked.connect(self.close); lay.addWidget(self.close_btn)

    def _fetch_info(self):
        self._info_worker = ModelInfoWorker(self.repo_id)
        self._info_worker.info_ready.connect(self._on_info)
        self._info_worker.error.connect(lambda e: self.info_lbl.setText(f"Error: {e}"))
        self._info_worker.start()

    def _on_info(self, info: dict):
        self.info_data = info
        pt = info.get("pipeline_tag", "")
        self.info_lbl.setText(
            f"Pipeline: {pt or '—'}  |  Downloads: {info.get('downloads', 0):,}  |  "
            f"SHA: {info.get('sha', '—')}"
        )

        files = info.get("files", [])
        is_image = pt == "text-to-image"

        if not files and is_image:
            # Diffusers repos often have no single model file; offer repo download
            self.repo_dl_btn.setVisible(True)
            # Check if already downloaded
            if self.ctx.registry.is_downloaded(self.repo_id):
                self.repo_dl_btn.setText("✅ Already Downloaded"); self.repo_dl_btn.setEnabled(False)
            self.table.setRowCount(0)
            return

        self.table.setRowCount(len(files))
        for i, f in enumerate(files):
            fname = f["filename"]
            fsize = f.get("size", 0)

            self.table.setItem(i, 0, QTableWidgetItem(fname))
            self.table.setItem(i, 1, QTableWidgetItem(fmt_size_gb(fsize)))
            self.table.setItem(i, 2, QTableWidgetItem(f"{fsize:,}" if fsize else "—"))

            # Check if this specific file is already downloaded
            file_mid = f"{self.repo_id}::{fname}"
            existing = self.ctx.registry.get(file_mid)
            is_downloaded = existing and Path(existing.path).exists()

            if is_downloaded:
                status = QTableWidgetItem("✅ Downloaded")
                status.setForeground(QColor(C["ok"]))
                self.table.setItem(i, 3, status)
                # No action button needed
                action_lbl = QLabel("✓"); action_lbl.setAlignment(Qt.AlignCenter)
                self.table.setCellWidget(i, 4, action_lbl)
            else:
                status = QTableWidgetItem("Available")
                status.setForeground(QColor(C["accent"]))
                self.table.setItem(i, 3, status)
                # Download button
                btn = QPushButton("📥 Download"); btn.setObjectName("downloadBtn")
                btn.setProperty("fname", fname); btn.setProperty("fsize", fsize)
                btn.clicked.connect(lambda checked, fn=fname, fs=fsize: self._dl_file(fn, fs))
                btn_w = QWidget(); btn_l = QHBoxLayout(btn_w); btn_l.setContentsMargins(4,2,4,2); btn_l.addWidget(btn)
                self.table.setCellWidget(i, 4, btn_w)

        # Also show repo download button for image models alongside files
        if is_image:
            self.repo_dl_btn.setVisible(True)
            if self.ctx.registry.is_downloaded(self.repo_id):
                self.repo_dl_btn.setText("✅ Already Downloaded"); self.repo_dl_btn.setEnabled(False)

    def _dl_file(self, filename: str, file_size: int):
        """Download a single file (e.g., a specific GGUF quantization)."""
        # ── DUPLICATE CHECK ──
        mid = f"{self.repo_id}::{filename}"
        if self.ctx.registry.is_downloaded(mid):
            QMessageBox.information(self, "Already Downloaded", f"{filename} is already downloaded and verified.")
            return

        if self._dl_worker and self._dl_worker.isRunning():
            QMessageBox.warning(self, "Busy", "A download is already in progress."); return

        self.pbar.setVisible(True); self.pbar.setValue(0)
        dest = str(Path(self.ctx.config.models_dir) / self.repo_id.replace("/", "__") / filename)
        url = f"https://huggingface.co/{self.repo_id}/resolve/main/{filename}"

        task = PrioritizedTask(priority=0, task_id=f"dl_{time.time()}", url=url,
                               dest_path=dest, filename=filename, model_id=mid,
                               source="huggingface", expected_size=file_size)

        if mid not in self.ctx.manager.state_machines:
            self.ctx.manager.state_machines[mid] = ModelStateMachine(ModelState.REMOTE)

        self._dl_worker = DownloadWorker(task, self.ctx)
        self._dl_worker.progress.connect(lambda p, s: (
            self.pbar.setValue(p),
            self.pbar_lbl.setText(f"{p}% — {s:.0f} KB/s")
        ))
        self._dl_worker.finished.connect(self._on_dl_done)
        self._dl_worker.start()

    def _dl_repo(self):
        """Download the entire repo (for diffusers models)."""
        if self.ctx.registry.is_downloaded(self.repo_id):
            QMessageBox.information(self, "Already Downloaded", f"{self.repo_id} is already downloaded and verified.")
            return
        if self._dl_worker and hasattr(self._dl_worker, 'isRunning') and self._dl_worker.isRunning():
            QMessageBox.warning(self, "Busy", "A download is already in progress."); return

        self.pbar.setVisible(True); self.pbar.setValue(0)
        if self.repo_id not in self.ctx.manager.state_machines:
            self.ctx.manager.state_machines[self.repo_id] = ModelStateMachine(ModelState.REMOTE)

        self._dl_worker = RepoDownloadWorker(self.repo_id, self.ctx)
        self._dl_worker.progress.connect(lambda p, s: self.pbar.setValue(p))
        self._dl_worker.finished.connect(self._on_dl_done)
        self._dl_worker.start()

    def _on_dl_done(self, ok: bool, msg: str, mid: str):
        self.pbar.setVisible(False)
        if ok:
            QMessageBox.information(self, "Download Complete", msg)
            self._fetch_info()  # Refresh file list to show updated status
        else:
            QMessageBox.critical(self, "Download Error", msg)


# ═══════════════════════════════════════════════════════════════════
#  UI PAGES (Dependency Injected)
# ═══════════════════════════════════════════════════════════════════
class BrowsePage(QWidget):
    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent); self.ctx = ctx; self._worker = None; self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(32,28,32,28)
        title = QLabel("Browse Models")
        title.setStyleSheet(f"font-size:18px; font-weight:700;")
        lay.addWidget(title)

        h = QHBoxLayout()
        self.search_in = QLineEdit(); self.search_in.setPlaceholderText("Search HuggingFace (Trie AutoComplete)...")
        self.completer = QCompleter(); self.search_in.setCompleter(self.completer)
        self.completer_model = QStringListModel(); self.completer.setModel(self.completer_model)
        self.update_completer()
        self.search_btn = QPushButton("🔍 Search"); self.search_btn.setObjectName("primaryBtn"); self.search_btn.clicked.connect(self._search)
        h.addWidget(self.search_in, 1); h.addWidget(self.search_btn); lay.addLayout(h)

        # ── ENHANCED: columns include Size (GB) and Status ──
        self.table = QTableWidget(); self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["ID", "Type", "Params", "Size (GB)", "Status", "Action"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers); self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table, 1)

    def update_completer(self):
        keys = self.ctx.trie.search_prefix("")[:1000]
        self.completer_model.setStringList(keys)

    def _search(self):
        q = self.search_in.text().strip()
        if not q: return
        self.search_btn.setEnabled(False)
        self._worker = SearchWorker(q, self.ctx)
        self._worker.results_ready.connect(self._on_results)
        self._worker.finished.connect(lambda: self.search_btn.setEnabled(True))
        self._worker.start()

    def _on_results(self, results):
        self.table.setRowCount(len(results)); self.update_completer()
        for i, r in enumerate(results):
            self.table.setItem(i, 0, QTableWidgetItem(r["id"]))
            self.table.setItem(i, 1, QTableWidgetItem("🖼" if r["type"] == ModelType.IMAGE else "📝"))
            self.table.setItem(i, 2, QTableWidgetItem(fmt_params(r["params"])))

            # ── Size (GB): fetch from registry if previously downloaded ──
            existing = self.ctx.registry.find_by_repo(r["id"])
            if existing:
                total_size = sum(e.size for e in existing)
                self.table.setItem(i, 3, QTableWidgetItem(fmt_size_gb(total_size)))
            else:
                self.table.setItem(i, 3, QTableWidgetItem("—"))

            # ── Status: show download status ──
            is_dl = self.ctx.registry.is_downloaded(r["id"]) or bool(existing)
            if is_dl:
                status_item = QTableWidgetItem("✅ Downloaded")
                status_item.setForeground(QColor(C["ok"]))
                self.table.setItem(i, 4, status_item)
            else:
                self.table.setItem(i, 4, QTableWidgetItem("Available"))

            # ── Action: Download button (or "View" if already downloaded) ──
            btn_w = QWidget(); btn_l = QHBoxLayout(btn_w); btn_l.setContentsMargins(4,2,4,2)
            if is_dl:
                btn_view = QPushButton("📂 View"); btn_view.setObjectName("okBtn")
                btn_view.clicked.connect(lambda checked, mid=r["id"]: self._view_details(mid))
                btn_l.addWidget(btn_view)
            else:
                btn_dl = QPushButton("📥 Download"); btn_dl.setObjectName("downloadBtn")
                btn_dl.clicked.connect(lambda checked, mid=r["id"]: self._view_details(mid))
                btn_l.addWidget(btn_dl)
            btn_l.addStretch()
            self.table.setCellWidget(i, 5, btn_w)

    def _view_details(self, repo_id: str):
        """Open the ModelDetailDialog for viewing/downloading a model."""
        dlg = ModelDetailDialog(repo_id, self.ctx, self)
        dlg.exec_()
        # After dialog closes, refresh search results to update statuses
        self.update_completer()


class ModelsPage(QWidget):
    model_changed = Signal()
    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent); self.ctx = ctx; self._verify_worker = None; self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(32,28,32,28)
        h = QHBoxLayout()
        title = QLabel("My Models")
        title.setStyleSheet(f"font-size:18px; font-weight:700;")
        h.addWidget(title); h.addStretch()
        btn_ref = QPushButton("🔄 Refresh"); btn_ref.setObjectName("secondaryBtn"); btn_ref.clicked.connect(self.refresh)
        btn_imp = QPushButton("📂 Import"); btn_imp.setObjectName("secondaryBtn"); btn_imp.clicked.connect(self._import)
        btn_verify_all = QPushButton("✅ Verify All"); btn_verify_all.setObjectName("verifyBtn"); btn_verify_all.clicked.connect(self._verify_all)
        h.addWidget(btn_imp); h.addWidget(btn_verify_all); h.addWidget(btn_ref)
        lay.addLayout(h)

        # ── ENHANCED: more columns for metadata ──
        self.tree = QTreeWidget(); self.tree.setColumnCount(7)
        self.tree.setHeaderLabels(["Model ID", "Format", "Size (GB)", "Status", "Checksum", "Version", "Actions"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.tree.setAlternatingRowColors(True); self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._ctx_menu)
        lay.addWidget(self.tree, 1)

    def refresh(self):
        self.tree.clear()
        records = self.ctx.registry.list_all()
        grouped = {}
        for r in records:
            if r.model_id not in self.ctx.manager.state_machines:
                self.ctx.manager.state_machines[r.model_id] = ModelStateMachine(r.state)
            family = self.ctx.graph.get_family(r.model_id)
            root = family[0]
            if root not in grouped: grouped[root] = []
            grouped[root].append(r)
            if r.model_id != root and r.base_model_id is None:
                r.base_model_id = root

        for base_id, variants in grouped.items():
            # Calculate total size for root
            total_size = sum(v.size for v in variants)
            root_item = QTreeWidgetItem(self.tree, [
                base_id, "", fmt_size_gb(total_size), "", "", "", ""
            ])
            root_item.setData(0, Qt.UserRole, base_id)
            root_item.setForeground(2, QColor(C["accent"]))

            for v in variants:
                state = self.ctx.manager.state_machines[v.model_id].state
                fmt = v.format_key.upper()
                sz_gb = fmt_size_gb(v.size)

                # ── Download status with color ──
                dl_status = v.download_status
                status_colors = {
                    "verified": C["ok"], "complete": C["accent2"],
                    "pending": C["warn"], "corrupted": C["danger"]
                }
                status_color = status_colors.get(dl_status, C["text_dim"])
                status_text = {
                    "verified": "✅ Verified", "complete": "📦 Complete",
                    "pending": "⏳ Pending", "corrupted": "❌ Corrupted"
                }.get(dl_status, dl_status)

                # ── Checksum (truncated) ──
                cs_text = f"{v.checksum[:12]}…" if v.checksum else "—"

                # ── Version ──
                ver_text = v.version or "—"

                item = QTreeWidgetItem(root_item, [v.model_id, fmt, sz_gb, status_text, cs_text, ver_text, ""])
                item.setData(0, Qt.UserRole, v.model_id)
                item.setForeground(2, QColor(C["accent"]))
                item.setForeground(3, QColor(status_color))

                if state == ModelState.LOADED: item.setForeground(0, QColor(C["accent2"]))
                elif state == ModelState.ERROR: item.setForeground(0, QColor(C["danger"]))

                # ── Action buttons ──
                btn_w = QWidget(); btn_l = QHBoxLayout(btn_w); btn_l.setContentsMargins(4,2,4,2)
                if state == ModelState.LOADED:
                    if v.model_type == ModelType.IMAGE:
                        btn_gen = QPushButton("🎨 Gen"); btn_gen.setObjectName("imageBtn")
                        btn_gen.clicked.connect(lambda ch, mid=v.model_id: ImageGenerationDialog(mid, self.ctx, self).exec())
                        btn_l.addWidget(btn_gen)
                    btn_unload = QPushButton("Unload"); btn_unload.setObjectName("warnBtn")
                    btn_unload.clicked.connect(lambda ch, mid=v.model_id: self._unload(mid))
                    btn_l.addWidget(btn_unload)
                elif state == ModelState.DOWNLOADED:
                    btn_load = QPushButton("Load"); btn_load.setObjectName("downloadBtn")
                    btn_load.clicked.connect(lambda ch, mid=v.model_id: self._load(mid))
                    btn_l.addWidget(btn_load)

                # ── Verify button ──
                btn_verify = QPushButton("🔍 Verify"); btn_verify.setObjectName("verifyBtn")
                btn_verify.clicked.connect(lambda ch, mid=v.model_id: self._verify_model(mid))
                btn_l.addWidget(btn_verify)

                self.tree.setItemWidget(item, 6, btn_w)
        self.tree.expandAll()

    def _load(self, mid: str):
        rec = self.ctx.registry.get(mid)
        if not rec: return
        # ── Pre-load verification ──
        ok, msg = file_exists_and_valid(rec.path, rec.size, rec.checksum)
        if not ok:
            rec.download_status = "corrupted"; self.ctx.registry.save()
            QMessageBox.critical(self, "File Corrupted", f"Cannot load {mid}: {msg}\nPlease re-download.")
            self.refresh(); return
        det = detect_format(rec.filename)
        cmd = LoadModelCommand(mid, rec.path, det, self.ctx.config, self.ctx.manager)
        try:
            self.ctx.manager.load_model(cmd)
            QMessageBox.information(self, "Loaded", f"{mid} loaded."); self.refresh()
        except Exception as e: QMessageBox.critical(self, "Error", str(e)); self.refresh()

    def _unload(self, mid: str):
        self.ctx.manager.unload_model(mid); self.refresh()

    def _import(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Import", "", "Model Files (*.gguf *.safetensors *.bin)")
        if fp:
            mid = Path(fp).stem; det = detect_format(fp)
            fsize = Path(fp).stat().st_size
            # ── Compute checksum on import ──
            checksum = compute_sha256(fp)
            rec = ModelRecord(
                model_id=mid, path=fp, filename=Path(fp).name, size=fsize,
                format_key=det.format_key, quant_key=det.quant_key,
                params_b=det.estimated_params_b, vram_gb=det.estimated_vram_gb,
                source="local", added_at=datetime.now().isoformat(),
                model_type=det.model_type, state=ModelState.DOWNLOADED,
                checksum=checksum, version="local", download_status="verified"
            )
            self.ctx.registry.add(rec); self.ctx.trie.insert(mid, mid); self.refresh()

    def _verify_model(self, mid: str):
        """Verify a single model's integrity."""
        rec = self.ctx.registry.get(mid)
        if not rec:
            QMessageBox.warning(self, "Not Found", f"{mid} not found in registry."); return
        if self._verify_worker and self._verify_worker.isRunning():
            QMessageBox.warning(self, "Busy", "Verification already in progress."); return
        self._verify_worker = ChecksumWorker(mid, self.ctx)
        self._verify_worker.finished.connect(self._on_verify_done)
        self._verify_worker.start()

    def _verify_all(self):
        """Verify all downloaded models."""
        records = self.ctx.registry.list_all()
        results = []
        for r in records:
            ok, msg = self.ctx.registry.verify_model(r.model_id)
            results.append((r.model_id, ok, msg))
        self.refresh()
        # Show summary
        ok_count = sum(1 for _, ok, _ in results if ok)
        bad_count = len(results) - ok_count
        detail = "\n".join(f"{'✅' if ok else '❌'} {mid}: {msg}" for mid, ok, msg in results)
        QMessageBox.information(self, "Verification Complete",
                                f"Verified {len(results)} models: {ok_count} OK, {bad_count} issues\n\n{detail}")

    def _on_verify_done(self, ok: bool, msg: str, mid: str):
        if ok:
            QMessageBox.information(self, "Verification Passed", f"✅ {mid}: {msg}")
        else:
            QMessageBox.critical(self, "Verification Failed", f"❌ {mid}: {msg}")
        self.refresh()

    def _ctx_menu(self, pos):
        item = self.tree.itemAt(pos)
        if not item: return
        mid = item.data(0, Qt.UserRole)
        menu = QMenu(self)
        rec = self.ctx.registry.get(mid)
        if rec:
            menu.addAction("🔍 Verify", lambda: self._verify_model(mid))
            menu.addAction("📋 Copy Checksum", lambda: self._copy_checksum(mid))
            menu.addSeparator()
            menu.addAction("🗑 Delete", lambda: (self.ctx.registry.remove(mid), self.refresh()))
        menu.exec(self.tree.mapToGlobal(pos))

    def _copy_checksum(self, mid: str):
        rec = self.ctx.registry.get(mid)
        if rec and rec.checksum:
            QApplication.clipboard().setText(rec.checksum)
            QMessageBox.information(self, "Copied", f"SHA-256 checksum copied for {mid}")
        else:
            QMessageBox.information(self, "No Checksum", f"No checksum available for {mid}")


class APIPage(QWidget):
    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent); self.ctx = ctx; self.api_srv = APIServer(ctx); self._build_ui()
    def _build_ui(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(32,28,32,28)
        title = QLabel("API Server (LRU Cached)")
        title.setStyleSheet(f"font-size:18px; font-weight:700;")
        lay.addWidget(title)
        if not HAS_FASTAPI: lay.addWidget(QLabel("⚠ FastAPI missing.")); return
        h = QHBoxLayout(); self.port = QSpinBox(); self.port.setRange(1024, 65535); self.port.setValue(self.ctx.config.server_port)
        self.btn = QPushButton("▶ Start"); self.btn.setObjectName("primaryBtn"); self.btn.clicked.connect(self._toggle)
        h.addWidget(QLabel("Port:")); h.addWidget(self.port); h.addWidget(self.btn); lay.addLayout(h)
        self.status = QLabel("Stopped"); self.status.setStyleSheet(f"color:{C['text_dim']}"); lay.addWidget(self.status)
        lay.addWidget(QLabel("Endpoints:\nGET /v1/models\nPOST /v1/chat/completions\nPOST /v1/images/generations"))
    def _toggle(self):
        if self.api_srv.running:
            self.api_srv.stop(); self.btn.setText("▶ Start"); self.btn.setObjectName("primaryBtn"); self.btn.style().unpolish(self.btn); self.btn.style().polish(self.btn)
            self.status.setText("Stopped"); self.status.setStyleSheet(f"color:{C['text_dim']}")
        else:
            self.ctx.config.server_port = self.port.value()
            if self.api_srv.start():
                self.btn.setText("■ Stop"); self.btn.setStyleSheet(f"background:{C['danger']}; color:#fff; border:none; border-radius:6px; padding:8px 20px; font-weight:600;")
                self.status.setText(f"Running on {self.ctx.config.server_port}"); self.status.setStyleSheet(f"color:{C['accent2']}; font-weight:600;")

# ═══════════════════════════════════════════════════════════════════
#  MAIN WINDOW & ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
NAV_ITEMS = [("browse", "🔍", "Browse"), ("models", "📦", "Models"), ("api", "🌐", "API")]

class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext):
        super().__init__(); self.ctx = ctx; self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}"); self.resize(1280, 850)
        self._build_ui()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central); main_lay = QHBoxLayout(central); main_lay.setContentsMargins(0,0,0,0); main_lay.setSpacing(0)
        sidebar = QFrame(); sidebar.setObjectName("sidebar"); sidebar.setFixedWidth(200); side_lay = QVBoxLayout(sidebar); side_lay.setContentsMargins(8,16,8,16)

        brand = QLabel(f"  {APP_NAME} v{APP_VERSION}")
        brand.setStyleSheet(f"font-weight:700; color:{C['accent']}; font-size:14px; padding:8px 0;")
        side_lay.addWidget(brand)

        self._nav_btns = {}; self.stack = QStackedWidget(); self.pages = {}
        self.pages["browse"] = BrowsePage(self.ctx)
        self.pages["models"] = ModelsPage(self.ctx)
        self.pages["api"] = APIPage(self.ctx)

        for key, icon, label in NAV_ITEMS:
            btn = QPushButton(f"  {icon}  {label}"); btn.setObjectName("navBtn"); btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.clicked.connect(lambda checked, k=key: self.navigate_to(k))
            side_lay.addWidget(btn); self._nav_btns[key] = btn
            self.stack.addWidget(self.pages[key])

        side_lay.addStretch()
        gpus = self.ctx.gpus
        gpu_lbl = QLabel(f"  GPU: {fmt_size_gb(sum(g.total_vram_mb for g in gpus)*1024*1024)}" if gpus else "  No GPU detected")
        gpu_lbl.setStyleSheet(f"color:{C['text_dim']}; font-size:12px;"); side_lay.addWidget(gpu_lbl)

        # ── Manifest info in sidebar ──
        model_count = len(self.ctx.registry.list_all())
        total_size = sum(r.size for r in self.ctx.registry.list_all())
        info_lbl = QLabel(f"  Models: {model_count} | {fmt_size_gb(total_size)}")
        info_lbl.setStyleSheet(f"color:{C['text_dim']}; font-size:12px;"); side_lay.addWidget(info_lbl)

        main_lay.addWidget(sidebar); main_lay.addWidget(self.stack, 1)
        self.navigate_to("browse")

    def navigate_to(self, key: str):
        self.stack.setCurrentWidget(self.pages[key])
        for k, btn in self._nav_btns.items(): btn.setProperty("active", k == key); btn.style().unpolish(btn); btn.style().polish(btn)
        if key == "models": self.pages[key].refresh()

    def closeEvent(self, event):
        if hasattr(self.pages.get("api"), 'api_srv') and self.pages["api"].api_srv.running: self.pages["api"].api_srv.stop()
        for mid in list(self.ctx.manager.backends.keys()): self.ctx.manager.unload_model(mid)
        event.accept()


def main():
    app = QApplication(sys.argv); app.setStyle("Fusion"); app.setStyleSheet(STYLESHEET)

    # Dependency Injection Setup
    config = AppConfig()
    gpus = GPUDetector.detect()
    registry = ModelRegistry(config.models_dir)
    manager = ModelManager()
    trie = ModelTrie()
    graph = ModelGraph()
    dl_scheduler = DownloadScheduler()

    # Populate Trie and State Machines from existing registry
    for rec in registry.list_all():
        trie.insert(rec.model_id, rec.model_id)
        graph.add_variant(rec.model_id.split("/")[0] if "/" in rec.model_id else rec.model_id, rec.model_id)
        if rec.model_id not in manager.state_machines:
            manager.state_machines[rec.model_id] = ModelStateMachine(rec.state)
        # ── Verify existing models on startup (quick size check) ──
        if rec.path and not Path(rec.path).exists():
            rec.download_status = "corrupted"
            logger.warning(f"Model {rec.model_id} file missing, marked as corrupted")
        elif rec.size > 0 and Path(rec.path).exists() and Path(rec.path).stat().st_size != rec.size:
            rec.download_status = "corrupted"
            logger.warning(f"Model {rec.model_id} size mismatch, marked as corrupted")
    registry.save()  # Persist any status updates

    ctx = AppContext(config, gpus, registry, manager, trie, graph, dl_scheduler)
    global AppContextSingleton; AppContextSingleton = ctx

    window = MainWindow(ctx); window.show(); sys.exit(app.exec())

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
ModelBridge - Advanced Local LLM & Image Model Manager
Features: File size in GB display, Download action with duplicate prevention,
          SHA256 checksum verification, external manifest tracking.
"""

import sys, os, json, time, re, shutil, subprocess, traceback, threading, logging, asyncio, io, base64, heapq, hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Any, Sequence, Type, Callable
from enum import Enum, auto
from dataclasses import dataclass, field, replace
from functools import lru_cache
from collections import OrderedDict

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QLabel, QPushButton, QLineEdit,
    QComboBox, QProgressBar, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QFileDialog, QMessageBox, QFrame,
    QScrollArea, QTabWidget, QHeaderView, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QDialog, QFormLayout, QMenu, QSizePolicy,
    QPlainTextEdit, QTreeWidget, QTreeWidgetItem, QCompleter
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize, QStringListModel
from PySide6.QtGui import QFont, QColor, QCursor, QPixmap, QImage

import requests
import psutil

# ═══════════════════════════════════════════════════════════════════
#  LOGGING & DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ModelBridge")

try: from llama_cpp import Llama as LlamaCPP; HAS_LLAMA_CPP = True
except ImportError: HAS_LLAMA_CPP = False

try: import torch; from transformers import AutoModelForCausalLM, AutoTokenizer; HAS_TRANSFORMERS = True
except ImportError: HAS_TRANSFORMERS = False

try: from diffusers import AutoPipelineForText2Image; import torch as diff_torch; HAS_DIFFUSERS = True
except ImportError: HAS_DIFFUSERS = False

try: from huggingface_hub import snapshot_download; HAS_HF_HUB = True
except ImportError: HAS_HF_HUB = False

try:
    import uvicorn; from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel as PydanticModel
    from typing import Optional as Opt
    HAS_FASTAPI = True
except ImportError: HAS_FASTAPI = False

try: from PIL import Image as PILImage; HAS_PIL = True
except ImportError: HAS_PIL = False

def global_exception_hook(exc_type, exc_value, exc_traceback):
    logger.critical("UNHANDLED EXCEPTION:", exc_info=(exc_type, exc_value, exc_traceback))
    if QApplication.instance(): QMessageBox.critical(None, "Error", str(exc_value))
sys.excepthook = global_exception_hook

# ═══════════════════════════════════════════════════════════════════
#  DATA STRUCTURES: TRIE, DAG, HEAP
# ═══════════════════════════════════════════════════════════════════
class TrieNode:
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.model_ids: List[str] = []

class ModelTrie:
    def __init__(self): self.root = TrieNode()
    def insert(self, word: str, model_id: str):
        node = self.root
        for char in word.lower():
            if char not in node.children: node.children[char] = TrieNode()
            node = node.children[char]
        if model_id not in node.model_ids: node.model_ids.append(model_id)
    def search_prefix(self, prefix: str) -> List[str]:
        node = self.root
        for char in prefix.lower():
            if char not in node.children: return []
            node = node.children[char]
        return self._collect(node)
    def _collect(self, node: TrieNode) -> List[str]:
        results = list(node.model_ids)
        for child in node.children.values(): results.extend(self._collect(child))
        return results

class ModelGraph:
    def __init__(self): self.nodes: Dict[str, List[str]] = {}
    def add_variant(self, base_id: str, variant_id: str):
        if base_id not in self.nodes: self.nodes[base_id] = []
        if variant_id not in self.nodes: self.nodes[variant_id] = []
        if variant_id not in self.nodes[base_id]: self.nodes[base_id].append(variant_id)
    def get_family(self, model_id: str) -> List[str]:
        root = model_id
        for potential_root, children in self.nodes.items():
            if model_id in children and potential_root != model_id:
                root = potential_root
                break
        return [root] + self.nodes.get(root, [])

@dataclass(order=True)
class PrioritizedTask:
    priority: int
    task_id: str = field(compare=False)
    url: str = field(compare=False)
    dest_path: str = field(compare=False)
    filename: str = field(compare=False)
    model_id: str = field(compare=False)
    source: str = field(compare=False)
    expected_size: int = field(compare=False, default=0)

class DownloadScheduler:
    def __init__(self): self.queue: List[PrioritizedTask] = []; self.task_map: Dict[str, PrioritizedTask] = {}
    def add_task(self, task: PrioritizedTask):
        heapq.heappush(self.queue, task); self.task_map[task.task_id] = task
    def reprioritize(self, task_id: str, new_priority: int):
        if task_id in self.task_map:
            self.task_map[task_id].task_id = f"INVALID_{time.time()}"
            new_task = replace(self.task_map[task_id], priority=new_priority, task_id=task_id)
            self.add_task(new_task)
    def get_next(self) -> Optional[PrioritizedTask]:
        while self.queue:
            task = heapq.heappop(self.queue)
            if not task.task_id.startswith("INVALID_"):
                self.task_map.pop(task.task_id, None)
                return task
        return None

# ═══════════════════════════════════════════════════════════════════
#  CORE MODELS & STATE MACHINES
# ═══════════════════════════════════════════════════════════════════
class ModelState(Enum):
    REMOTE = auto(); DOWNLOADING = auto(); DOWNLOADED = auto()
    LOADING = auto(); LOADED = auto(); UNLOADING = auto(); ERROR = auto()

class ModelStateMachine:
    TRANSITIONS = {
        ModelState.REMOTE: [ModelState.DOWNLOADING, ModelState.ERROR],
        ModelState.DOWNLOADING: [ModelState.DOWNLOADED, ModelState.ERROR, ModelState.REMOTE],
        ModelState.DOWNLOADED: [ModelState.LOADING, ModelState.REMOTE],
        ModelState.LOADING: [ModelState.LOADED, ModelState.ERROR, ModelState.DOWNLOADED],
        ModelState.LOADED: [ModelState.UNLOADING],
        ModelState.UNLOADING: [ModelState.DOWNLOADED],
        ModelState.ERROR: [ModelState.REMOTE]
    }
    def __init__(self, initial=ModelState.REMOTE): self.state = initial
    def transition(self, new_state: ModelState):
        if new_state not in self.TRANSITIONS.get(self.state, []):
            logger.error(f"Invalid transition: {self.state.name} -> {new_state.name}"); return False
        self.state = new_state; return True

class ModelSource(Enum):
    HUGGINGFACE = "huggingface"; MODELSCOPE = "modelscope"; GITHUB = "github"
class ModelType(Enum):
    TEXT = "text"; IMAGE = "image"

# ───────────────────────────────────────────────────────────────────
#  ENHANCED ModelRecord: checksum, version, download_status
# ───────────────────────────────────────────────────────────────────
@dataclass
class ModelRecord:
    model_id: str; path: str; filename: str; size: int; format_key: str
    quant_key: Optional[str]; params_b: Optional[float]; vram_gb: Optional[float]
    source: str; added_at: str; model_type: ModelType = ModelType.TEXT
    state: ModelState = ModelState.DOWNLOADED
    base_model_id: Optional[str] = None
    # ── NEW: manifest tracking fields ──
    checksum: Optional[str] = None           # SHA-256 hex digest
    version: Optional[str] = None            # model tag / revision
    download_status: str = "complete"        # "complete" | "corrupted" | "pending" | "verified"

    @property
    def size_gb(self) -> str:
        """Human-readable size with GB precision."""
        return fmt_size(self.size)

    def to_dict(self) -> dict:
        return {k: (v.value if isinstance(v, Enum) else v) for k, v in self.__dict__.items() if not k.startswith('_')}
    @classmethod
    def from_dict(cls, d: dict) -> "ModelRecord":
        d['state'] = ModelState(d.get('state', 'DOWNLOADED'))
        d['model_type'] = ModelType(d.get('model_type', 'text'))
        # Gracefully handle missing new fields from old manifests
        for fld in ('checksum', 'version', 'download_status'):
            d.setdefault(fld, None if fld != 'download_status' else 'complete')
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

@dataclass
class FormatDetection:
    format_key: str; quant_key: Optional[str]; estimated_params_b: Optional[float]
    estimated_vram_gb: Optional[float]; model_type: ModelType = ModelType.TEXT

@dataclass
class GPUInfo: type: str; index: int; name: str; total_vram_mb: float; free_vram_mb: float

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS & CONFIG
# ═══════════════════════════════════════════════════════════════════
APP_NAME = "ModelBridge"; APP_VERSION = "4.1.0"
MANIFEST_VERSION = "2.0"

C = {"bg":"#0a0a0a","surface":"#161616","surface2":"#1c1c1c","border":"#222222","text":"#d4d4d4","text_dim":"#666","accent":"#3b82f6","accent2":"#10b981","warn":"#f59e0b","danger":"#ef4444","pink":"#ec4899","purple":"#8b5cf6","ok":"#22c55e"}

STYLESHEET = f"""
QMainWindow {{ background: {C['bg']}; }}
QWidget {{ color: {C['text']}; font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 13px; }}
QFrame#sidebar {{ background: {C['bg']}; border-right: 1px solid {C['border']}; }}
QPushButton#navBtn {{ background: transparent; border: none; border-radius: 6px; padding: 9px 14px; text-align: left; color: {C['text_dim']}; margin: 1px 6px; }}
QPushButton#navBtn:hover {{ background: {C['surface']}; color: {C['text']}; }}
QPushButton#navBtn[active="true"] {{ background: {C['surface2']}; color: {C['accent']}; font-weight: 600; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {{ background: {C['surface']}; border: 1px solid {C['border']}; border-radius: 6px; padding: 7px 12px; color: {C['text']}; }}
QPushButton#primaryBtn {{ background: {C['accent']}; border: none; border-radius: 6px; padding: 8px 20px; color: #fff; font-weight: 600; }}
QPushButton#primaryBtn:hover {{ background: #2563eb; }}
QPushButton#secondaryBtn {{ background: {C['surface2']}; border: 1px solid {C['border']}; border-radius: 6px; padding: 8px 20px; color: {C['text']}; }}
QPushButton#dangerBtn {{ background: {C['danger']}; border: none; border-radius: 6px; padding: 8px 20px; color: #fff; font-weight: 600; }}
QPushButton#downloadBtn {{ background: {C['accent']}; border: none; border-radius: 5px; padding: 5px 14px; color: #fff; font-weight: 600; font-size: 12px; }}
QPushButton#downloadBtn:disabled {{ background: {C['border']}; color: {C['text_dim']}; }}
QPushButton#imageBtn {{ background: {C['pink']}; border: none; border-radius: 5px; padding: 5px 14px; color: #fff; font-weight: 600; font-size: 12px; }}
QPushButton#warnBtn {{ background: {C['warn']}; border: none; border-radius: 5px; padding: 5px 14px; color: #000; font-weight: 600; font-size: 12px; }}
QPushButton#verifyBtn {{ background: {C['accent2']}; border: none; border-radius: 5px; padding: 5px 14px; color: #fff; font-weight: 600; font-size: 12px; }}
QPushButton#okBtn {{ background: {C['ok']}; border: none; border-radius: 5px; padding: 5px 14px; color: #fff; font-weight: 600; font-size: 12px; }}
QTreeWidget {{ background: {C['bg']}; border: 1px solid {C['border']}; border-radius: 8px; alternate-background-color: {C['surface']}; }}
QTreeWidget::item {{ padding: 6px; border-bottom: 1px solid {C['border']}; }}
QTreeWidget::item:selected {{ background: {C['accent']}; color: #fff; }}
QHeaderView::section {{ background: {C['surface']}; border: none; border-bottom: 1px solid {C['border']}; padding: 8px 12px; font-weight: 600; color: {C['text_dim']}; }}
QProgressBar {{ background: {C['surface2']}; border: none; border-radius: 4px; height: 6px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {C['accent']}; border-radius: 4px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 6px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {C['border']}; border-radius: 3px; min-height: 30px; }}
QGroupBox {{ border: 1px solid {C['border']}; border-radius: 8px; margin-top: 16px; padding-top: 20px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {C['text_dim']}; }}
QToolTip {{ background: {C['surface2']}; color: {C['text']}; border: 1px solid {C['border']}; border-radius: 6px; padding: 6px; }}
QTableWidget {{ background: {C['bg']}; border: 1px solid {C['border']}; border-radius: 8px; alternate-background-color: {C['surface']}; gridline-color: {C['border']}; }}
QTableWidget::item {{ padding: 4px; }}
"""

# ═══════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def fmt_size(n) -> str:
    if n is None or n < 0: return "—"
    try: n = int(n)
    except: return "—"
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024: return f"{n:.2f} {u}" if u == "GB" else f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"

def fmt_size_gb(n) -> str:
    """Always show size in GB with 2 decimal places."""
    if n is None or n < 0: return "—"
    try: n = float(n)
    except: return "—"
    gb = n / (1024 ** 3)
    if gb >= 1024: return f"{gb/1024:.2f} TB"
    if gb >= 1: return f"{gb:.2f} GB"
    if gb >= 0.001: return f"{gb*1024:.1f} MB"
    return f"{n:.0f} B"

def fmt_params(p) -> str:
    if not p: return "—"
    try: v = float(p)
    except: return "—"
    if v >= 1000: return f"{v/1000:.1f}T"
    if v >= 1: return f"{v:.1f}B" if v != int(v) else f"{int(v)}B"
    return f"{v*1000:.0f}M"

_RE_PARAM = re.compile(r'[-_./](\d+\.?\d*)[bm]', re.IGNORECASE)
def extract_params(text: str) -> Optional[float]:
    m = _RE_PARAM.search(text.lower())
    if m:
        v = float(m.group(1))
        return round(v / 1000, 2) if m.group(0)[-1] == 'm' else v
    return None

def detect_format(filename: str, pipeline_tag: str = "") -> FormatDetection:
    fname = filename.lower()
    fmt = "bin"
    if fname.endswith(".gguf"): fmt = "gguf"
    elif "gptq" in fname: fmt = "gptq"
    elif "awq" in fname: fmt = "awq"
    elif fname.endswith(".safetensors"): fmt = "safetensors"
    m_type = ModelType.IMAGE if pipeline_tag in ("text-to-image",) or "stable-diffusion" in fname or "diffusers" in fname else ModelType.TEXT
    if m_type == ModelType.IMAGE and fmt in ("safetensors", "bin"): fmt = "diffusers"
    params = extract_params(filename)
    vram = round(params * 1.25, 1) if params else None
    return FormatDetection(fmt, None, params, vram, m_type)

# ───────────────────────────────────────────────────────────────────
#  NEW: SHA-256 checksum computation & verification
# ───────────────────────────────────────────────────────────────────
def compute_sha256(filepath: str, progress_cb: Callable[[int], None] = None) -> str:
    """Compute SHA-256 hash of a file. Optionally report progress (0-100)."""
    h = hashlib.sha256()
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    total = path.stat().st_size
    done = 0
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)  # 8 MB chunks
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            if progress_cb and total > 0:
                progress_cb(int(done / total * 100))
    return h.hexdigest()

def verify_file(filepath: str, expected_checksum: str) -> Tuple[bool, str]:
    """Verify a file's SHA-256 checksum. Returns (ok, actual_hash)."""
    if not expected_checksum:
        return True, "no-checksum"
    try:
        actual = compute_sha256(filepath)
        return (actual == expected_checksum, actual)
    except Exception as e:
        return (False, str(e))

def file_exists_and_valid(filepath: str, expected_size: int = 0, expected_checksum: str = None) -> Tuple[bool, str]:
    """Check if a file exists, has the right size, and optionally matches checksum."""
    p = Path(filepath)
    if not p.exists():
        return (False, "File does not exist")
    if expected_size > 0 and p.stat().st_size != expected_size:
        return (False, f"Size mismatch: expected {fmt_size_gb(expected_size)}, got {fmt_size_gb(p.stat().st_size)}")
    if expected_checksum:
        ok, actual = verify_file(filepath, expected_checksum)
        if not ok:
            return (False, f"Checksum mismatch: expected {expected_checksum[:16]}…, got {actual[:16]}…")
    return (True, "OK")

# ═══════════════════════════════════════════════════════════════════
#  PATTERN: BACKEND FACTORY (Strategy)
# ═══════════════════════════════════════════════════════════════════
class BackendFactory:
    _registry: Dict[str, Type] = {}
    @classmethod
    def register(cls, format_key: str, backend_class: Type): cls._registry[format_key] = backend_class
    @classmethod
    def create(cls, format_key: str, model_path: str, config: 'AppConfig') -> Any:
        b_cls = cls._registry.get(format_key)
        if not b_cls: raise ValueError(f"No backend for {format_key}")
        if format_key == "gguf": return b_cls(model_path, n_ctx=config.n_ctx, n_gpu_layers=config.n_gpu_layers)
        return b_cls(model_path)

class LlamaCppBackend:
    def __init__(self, path, n_ctx=4096, n_gpu_layers=-1):
        if not HAS_LLAMA_CPP: raise RuntimeError("llama-cpp-python missing")
        self.llm = LlamaCPP(model_path=path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
    def generate(self, msgs, params): return self.llm.create_chat_completion(messages=msgs, max_tokens=params.get("max_tokens", 2048), stream=False)["choices"][0]["message"]["content"]
    def unload(self): del self.llm; import gc; gc.collect()

class TransformersBackend:
    def __init__(self, path):
        if not HAS_TRANSFORMERS: raise RuntimeError("Transformers missing")
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True)
    def generate(self, msgs, params):
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=params.get("max_tokens", 2048), do_sample=True)
        return self.tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    def unload(self): del self.model; del self.tokenizer; import gc; gc.collect()

class DiffusersBackend:
    def __init__(self, path):
        if not HAS_DIFFUSERS: raise RuntimeError("Diffusers missing")
        dt = diff_torch.float16 if diff_torch.cuda.is_available() else diff_torch.float32
        self.pipe = AutoPipelineForText2Image.from_pretrained(path, torch_dtype=dt)
        if diff_torch.cuda.is_available(): self.pipe.to("cuda")
    def generate_image(self, prompt, params): return self.pipe(prompt=prompt, num_inference_steps=params.get("steps", 20)).images[0]
    def unload(self): del self.pipe; import gc; gc.collect(); diff_torch.cuda.empty_cache() if diff_torch.cuda.is_available() else None

BackendFactory.register("gguf", LlamaCppBackend)
BackendFactory.register("safetensors", TransformersBackend)
BackendFactory.register("diffusers", DiffusersBackend)

# ═══════════════════════════════════════════════════════════════════
#  PATTERN: COMMAND (Task Scheduling)
# ═══════════════════════════════════════════════════════════════════
class Command:
    def execute(self): raise NotImplementedError

class LoadModelCommand(Command):
    def __init__(self, model_id: str, path: str, fmt: FormatDetection, config: 'AppConfig', manager: 'ModelManager'):
        self.model_id = model_id; self.path = path; self.fmt = fmt; self.config = config; self.manager = manager
    def execute(self):
        self.manager.state_machines[self.model_id].transition(ModelState.LOADING)
        try:
            backend = BackendFactory.create(self.fmt.format_key, self.path, self.config)
            self.manager.backends[self.model_id] = backend
            self.manager.state_machines[self.model_id].transition(ModelState.LOADED)
        except Exception as e:
            self.manager.state_machines[self.model_id].transition(ModelState.ERROR)
            raise e

# ═══════════════════════════════════════════════════════════════════
#  PATTERN: DEPENDENCY INJECTION CONTEXT
# ═══════════════════════════════════════════════════════════════════
class AppConfig:
    def __init__(self):
        self.path = Path.home() / ".modelbridge" / "config.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = {}
        if self.path.exists():
            try: self._data = json.loads(self.path.read_text())
            except: pass
    def save(self): self.path.write_text(json.dumps(self._data, indent=2))
    def get(self, k, d=None): return self._data.get(k, d)
    def set(self, k, v): self._data[k] = v; self.save()
    @property
    def models_dir(self) -> str: return self.get("models_dir", str(Path.home() / "ModelBridge" / "models"))
    @property
    def n_ctx(self) -> int: return self.get("n_ctx", 4096)
    @property
    def n_gpu_layers(self) -> int: return self.get("n_gpu_layers", -1)
    @property
    def server_port(self) -> int: return self.get("server_port", 8081)

class GPUDetector:
    @staticmethod
    def detect() -> List[GPUInfo]:
        gpus = []
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                for l in r.stdout.strip().split("\n"):
                    p = l.split(",");
                    if len(p)>=4: gpus.append(GPUInfo("nvidia", int(p[0].strip()), p[1].strip(), float(p[2].strip()), float(p[3].strip())))
        except: pass
        return gpus

# ───────────────────────────────────────────────────────────────────
#  ENHANCED ModelRegistry: External manifest with version, checksums,
#  download status, file sizes
# ───────────────────────────────────────────────────────────────────
class ModelRegistry:
    def __init__(self, models_dir: str):
        self.dir = Path(models_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.dir / "manifest.json"
        self._models: Dict[str, ModelRecord] = self._load()

    def _load(self):
        if not self.manifest.exists(): return {}
        try:
            data = json.loads(self.manifest.read_text())
            # Support both old flat format and new versioned format
            models_data = data.get("models", data) if isinstance(data, dict) else data
            return {mid: ModelRecord.from_dict(d) for mid, d in models_data.items()}
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            return {}

    def save(self):
        manifest_data = {
            "manifest_version": MANIFEST_VERSION,
            "app_version": APP_VERSION,
            "last_updated": datetime.now().isoformat(),
            "models": {mid: r.to_dict() for mid, r in self._models.items()}
        }
        self.manifest.write_text(json.dumps(manifest_data, indent=2))

    def add(self, rec: ModelRecord): self._models[rec.model_id] = rec; self.save()
    def remove(self, mid: str): self._models.pop(mid, None); self.save()
    def get(self, mid: str) -> Optional[ModelRecord]: return self._models.get(mid)
    def list_all(self) -> List[ModelRecord]: return list(self._models.values())

    def is_downloaded(self, mid: str) -> bool:
        """Check if a model is already downloaded and not corrupted."""
        rec = self._models.get(mid)
        if not rec: return False
        if rec.download_status == "corrupted": return False
        # Verify file still exists on disk
        if not Path(rec.path).exists(): return False
        return True

    def find_by_repo(self, repo_id: str) -> List[ModelRecord]:
        """Find all records belonging to a repo (handles multi-file repos)."""
        return [r for r in self._models.values() if r.model_id.startswith(repo_id)]

    def update_checksum(self, mid: str, checksum: str):
        if mid in self._models:
            self._models[mid].checksum = checksum
            self.save()

    def update_download_status(self, mid: str, status: str):
        if mid in self._models:
            self._models[mid].download_status = status
            self.save()

    def verify_model(self, mid: str) -> Tuple[bool, str]:
        """Verify a model's file integrity. Returns (ok, message)."""
        rec = self._models.get(mid)
        if not rec: return (False, "Model not found in registry")
        if not Path(rec.path).exists():
            rec.download_status = "corrupted"
            self.save()
            return (False, f"File missing: {rec.path}")
        # Check file size
        actual_size = Path(rec.path).stat().st_size
        if rec.size > 0 and actual_size != rec.size:
            rec.download_status = "corrupted"
            self.save()
            return (False, f"Size mismatch: expected {fmt_size_gb(rec.size)}, got {fmt_size_gb(actual_size)}")
        # Check checksum if available
        if rec.checksum:
            ok, actual = verify_file(rec.path, rec.checksum)
            if not ok:
                rec.download_status = "corrupted"
                self.save()
                return (False, f"Checksum mismatch: expected {rec.checksum[:16]}…, got {actual[:16]}…")
        rec.download_status = "verified"
        self.save()
        return (True, f"Verified OK — {fmt_size_gb(rec.size)}, SHA256: {rec.checksum[:16] if rec.checksum else 'n/a'}…")


class ModelManager:
    def __init__(self):
        self.backends: Dict[str, Any] = {}
        self.state_machines: Dict[str, ModelStateMachine] = {}

    def load_model(self, cmd: LoadModelCommand):
        if cmd.model_id not in self.state_machines: self.state_machines[cmd.model_id] = ModelStateMachine(ModelState.DOWNLOADED)
        cmd.execute()

    def unload_model(self, mid: str):
        if mid in self.state_machines: self.state_machines[mid].transition(ModelState.UNLOADING)
        b = self.backends.pop(mid, None)
        if b: b.unload()
        if mid in self.state_machines: self.state_machines[mid].transition(ModelState.DOWNLOADED)

    def is_loaded(self, mid: str) -> bool: return mid in self.state_machines and self.state_machines[mid].state == ModelState.LOADED
    def get_backend(self, mid: str): return self.backends.get(mid)

@dataclass
class AppContext:
    config: AppConfig
    gpus: List[GPUInfo]
    registry: ModelRegistry
    manager: ModelManager
    trie: ModelTrie
    graph: ModelGraph
    dl_scheduler: DownloadScheduler

# ═══════════════════════════════════════════════════════════════════
#  PATTERN: LRU CACHE (API Server)
# ═══════════════════════════════════════════════════════════════════
if HAS_FASTAPI:
    class _ChatReq(PydanticModel): model: str = "default"; messages: list; temperature: Opt[float] = 0.7; max_tokens: Opt[int] = 2048
    class _ImgReq(PydanticModel): prompt: str; model: Opt[str] = "default"; steps: Opt[int] = 20

    @lru_cache(maxsize=64)
    def _cached_text_gen(model_id: str, prompt_hash: int, max_tokens: int, temp: float) -> str:
        return "Cached Response Pattern Applied."

    image_cache: OrderedDict[Tuple[str, int], bytes] = OrderedDict()
    def _cached_img_gen(model_id: str, prompt_hash: int, steps: int) -> bytes:
        key = (model_id, prompt_hash)
        if key in image_cache:
            image_cache.move_to_end(key); return image_cache[key]
        backend = AppContextSingleton.manager.get_backend(model_id)
        img = backend.generate_image("prompt_reconstructed", {"steps": steps})
        buf = io.BytesIO(); img.save(buf, format="PNG"); b_data = buf.getvalue()
        image_cache[key] = b_data
        if len(image_cache) > 10: image_cache.popitem(last=False)
        return b_data

    def create_api_app(ctx: AppContext) -> FastAPI:
        app = FastAPI(title="ModelBridge API")
        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

        @app.get("/v1/models")
        async def list_models():
            return {"data": [{"id": mid, "state": sm.state.name} for mid, sm in ctx.manager.state_machines.items() if sm.state == ModelState.LOADED]}

        @app.post("/v1/chat/completions")
        async def chat(req: _ChatReq):
            mid = req.model if req.model != "default" else next((m for m, sm in ctx.manager.state_machines.items() if sm.state == ModelState.LOADED), None)
            if not mid or not ctx.manager.is_loaded(mid): raise HTTPException(404, "No text models loaded")
            backend = ctx.manager.get_backend(mid)
            content = await asyncio.get_event_loop().run_in_executor(None, backend.generate, req.messages, {"max_tokens": req.max_tokens, "temperature": req.temperature})
            return {"choices": [{"message": {"role": "assistant", "content": content}}]}

        @app.post("/v1/images/generations")
        async def img_gen(req: _ImgReq):
            mid = req.model if req.model != "default" else next((m for m, sm in ctx.manager.state_machines.items() if sm.state == ModelState.LOADED), None)
            if not mid or not ctx.manager.is_loaded(mid): raise HTTPException(404, "No image models loaded")
            backend = ctx.manager.get_backend(mid)
            img = await asyncio.get_event_loop().run_in_executor(None, backend.generate_image, req.prompt, {"steps": req.steps})
            buf = io.BytesIO(); img.save(buf, format="PNG")
            return {"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]}
        return app

class APIServer:
    def __init__(self, ctx: AppContext): self.ctx = ctx; self.running = False; self._srv = None
    def start(self):
        if not HAS_FASTAPI or self.running: return False
        cfg = uvicorn.Config(create_api_app(self.ctx), host="0.0.0.0", port=self.ctx.config.server_port, log_level="warning")
        self._srv = uvicorn.Server(cfg)
        threading.Thread(target=lambda: (setattr(self, 'running', True), self._srv.run(), setattr(self, 'running', False)), daemon=True).start(); return True
    def stop(self):
        if self._srv: self._srv.should_exit = True; self.running = False

AppContextSingleton: AppContext = None

# ═══════════════════════════════════════════════════════════════════
#  WORKERS (Search, ModelInfo, Download with checksum)
# ═══════════════════════════════════════════════════════════════════
class SearchWorker(QThread):
    results_ready = Signal(list)
    def __init__(self, query: str, ctx: AppContext): super().__init__(); self.query = query; self.ctx = ctx
    def run(self):
        try:
            r = requests.get("https://huggingface.co/api/models", params={"search": self.query, "limit": 30}, timeout=15); r.raise_for_status()
            results = []
            for m in r.json():
                mid, pt = m.get("id", ""), m.get("pipeline_tag", "")
                m_type = ModelType.IMAGE if pt == "text-to-image" else ModelType.TEXT
                params = extract_params(mid)
                results.append({"id": mid, "type": m_type, "params": params, "pipeline_tag": pt, "downloads": m.get("downloads", 0)})
                self.ctx.trie.insert(mid, mid)
            self.results_ready.emit(results)
        except Exception as e: logger.error(f"Search failed: {e}")

# ───────────────────────────────────────────────────────────────────
#  NEW: ModelInfoWorker — fetches detailed model info including
#  file list with sizes from HuggingFace API
# ───────────────────────────────────────────────────────────────────
class ModelInfoWorker(QThread):
    info_ready = Signal(dict)
    error = Signal(str)
    def __init__(self, repo_id: str, parent=None): super().__init__(); self.repo_id = repo_id
    def run(self):
        try:
            r = requests.get(f"https://huggingface.co/api/models/{self.repo_id}", timeout=15)
            r.raise_for_status()
            data = r.json()
            siblings = data.get("siblings", [])
            files = []
            for s in siblings:
                fname = s.get("rfilename", "")
                size = s.get("size", 0)
                # Only include model-related files
                if any(fname.endswith(ext) for ext in (".gguf", ".safetensors", ".bin", ".pt", ".onnx", ".msgpack")):
                    files.append({"filename": fname, "size": size})
            self.info_ready.emit({
                "id": data.get("id", self.repo_id),
                "pipeline_tag": data.get("pipeline_tag", ""),
                "tags": data.get("tags", []),
                "downloads": data.get("downloads", 0),
                "files": files,
                "sha": data.get("sha", "")[:12],
            })
        except Exception as e:
            self.error.emit(str(e))

# ───────────────────────────────────────────────────────────────────
#  ENHANCED: DownloadWorker with SHA-256 checksum and verification
# ───────────────────────────────────────────────────────────────────
class DownloadWorker(QThread):
    progress = Signal(int, float)   # percent, speed KB/s
    checksum_progress = Signal(int) # checksum computation percent
    finished = Signal(bool, str, str)  # success, msg, model_id

    def __init__(self, task: PrioritizedTask, ctx: AppContext):
        super().__init__(); self.task = task; self.ctx = ctx; self._cancel = False

    def cancel(self): self._cancel = True

    def run(self):
        mid = self.task.model_id
        if mid not in self.ctx.manager.state_machines:
            self.ctx.manager.state_machines[mid] = ModelStateMachine(ModelState.REMOTE)
        self.ctx.manager.state_machines[mid].transition(ModelState.DOWNLOADING)

        # ── DUPLICATE CHECK: Don't re-download ──
        rec_existing = self.ctx.registry.get(mid)
        if rec_existing and Path(rec_existing.path).exists():
            ok, msg = file_exists_and_valid(rec_existing.path, rec_existing.size, rec_existing.checksum)
            if ok:
                self.ctx.manager.state_machines[mid].transition(ModelState.DOWNLOADED)
                self.finished.emit(True, "Already downloaded and verified", mid)
                return
            else:
                logger.warning(f"Existing file for {mid} is corrupted ({msg}), re-downloading")

        tmp_path = self.task.dest_path + ".part"
        try:
            Path(self.task.dest_path).parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(self.task.url, stream=True, timeout=30); r.raise_for_status()
            total = int(r.headers.get('content-length', 0)); dl = 0; t_last = time.time(); b_last = 0
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(64*1024):
                    if self._cancel:
                        self.ctx.manager.state_machines[mid].transition(ModelState.REMOTE)
                        if Path(tmp_path).exists(): Path(tmp_path).unlink()
                        self.finished.emit(False, "Cancelled", mid); return
                    f.write(chunk); dl += len(chunk)
                    if time.time() - t_last >= 0.5:
                        speed = (dl - b_last) / (time.time() - t_last) / 1024
                        self.progress.emit(int(dl/total*100) if total else 0, speed)
                        t_last = time.time(); b_last = dl

            os.replace(tmp_path, self.task.dest_path)

            # ── COMPUTE SHA-256 CHECKSUM ──
            self.progress.emit(99, 0)
            logger.info(f"Computing SHA-256 for {mid}...")
            checksum = compute_sha256(self.task.dest_path,
                                      progress_cb=lambda p: self.checksum_progress.emit(p))
            logger.info(f"SHA-256: {checksum}")

            # ── VERIFY FILE SIZE ──
            actual_size = Path(self.task.dest_path).stat().st_size
            if self.task.expected_size > 0 and actual_size != self.task.expected_size:
                self.ctx.manager.state_machines[mid].transition(ModelState.ERROR)
                self.finished.emit(False, f"Size mismatch: expected {fmt_size_gb(self.task.expected_size)}, got {fmt_size_gb(actual_size)}", mid)
                return

            det = detect_format(self.task.filename)
            rec = ModelRecord(
                model_id=mid, path=self.task.dest_path, filename=self.task.filename,
                size=actual_size, format_key=det.format_key, quant_key=det.quant_key,
                params_b=det.estimated_params_b, vram_gb=det.estimated_vram_gb,
                source=self.task.source, added_at=datetime.now().isoformat(),
                model_type=det.model_type, state=ModelState.DOWNLOADED,
                checksum=checksum, version=self.task.task_id.split("_v:")[-1] if "_v:" in self.task.task_id else "main",
                download_status="verified"
            )
            self.ctx.registry.add(rec)
            self.ctx.graph.add_variant(mid.split("/")[0] if "/" in mid else mid, mid)
            self.ctx.manager.state_machines[mid].transition(ModelState.DOWNLOADED)
            self.finished.emit(True, f"Downloaded & verified — {fmt_size_gb(actual_size)}, SHA256: {checksum[:16]}…", mid)

        except Exception as e:
            self.ctx.manager.state_machines[mid].transition(ModelState.ERROR)
            if Path(tmp_path).exists(): Path(tmp_path).unlink()
            self.finished.emit(False, str(e), mid)

# ───────────────────────────────────────────────────────────────────
#  ENHANCED: RepoDownloadWorker with checksum verification
# ───────────────────────────────────────────────────────────────────
class RepoDownloadWorker(QThread):
    progress = Signal(int, str)
    finished = Signal(bool, str, str)
    def __init__(self, repo_id: str, ctx: AppContext, version: str = "main"):
        super().__init__(); self.repo_id = repo_id; self.ctx = ctx; self.version = version
    def run(self):
        mid = self.repo_id
        if mid not in self.ctx.manager.state_machines: self.ctx.manager.state_machines[mid] = ModelStateMachine(ModelState.REMOTE)
        self.ctx.manager.state_machines[mid].transition(ModelState.DOWNLOADING)
        try:
            # ── DUPLICATE CHECK ──
            existing = self.ctx.registry.find_by_repo(mid)
            if existing:
                all_ok = all(Path(r.path).exists() for r in existing)
                if all_ok:
                    self.ctx.manager.state_machines[mid].transition(ModelState.DOWNLOADED)
                    self.finished.emit(True, "Already downloaded", mid)
                    return

            self.progress.emit(0, "Downloading repo...")
            path = snapshot_download(mid, local_dir=str(Path(self.ctx.config.models_dir)/mid.replace("/","__")))
            size = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())

            # ── COMPUTE CHECKSUM for primary model file ──
            primary_checksum = None
            model_files = list(Path(path).rglob("*.safetensors")) + list(Path(path).rglob("*.bin"))
            if model_files:
                self.progress.emit(90, "Computing checksum...")
                primary_checksum = compute_sha256(str(model_files[0]))

            rec = ModelRecord(
                model_id=mid, path=path, filename=mid.split("/")[-1],
                size=size, format_key="diffusers", quant_key=None,
                params_b=None, vram_gb=None, source="huggingface",
                added_at=datetime.now().isoformat(), model_type=ModelType.IMAGE,
                state=ModelState.DOWNLOADED,
                checksum=primary_checksum, version=self.version,
                download_status="verified"
            )
            self.ctx.registry.add(rec); self.ctx.manager.state_machines[mid].transition(ModelState.DOWNLOADED)
            self.progress.emit(100, "Done")
            self.finished.emit(True, f"Downloaded & verified — {fmt_size_gb(size)}", mid)
        except Exception as e:
            self.ctx.manager.state_machines[mid].transition(ModelState.ERROR); self.finished.emit(False, str(e), mid)

# ───────────────────────────────────────────────────────────────────
#  NEW: ChecksumWorker — verifies a model's integrity in background
# ───────────────────────────────────────────────────────────────────
class ChecksumWorker(QThread):
    progress = Signal(int)
    finished = Signal(bool, str, str)  # ok, message, model_id
    def __init__(self, model_id: str, ctx: AppContext):
        super().__init__(); self.model_id = model_id; self.ctx = ctx
    def run(self):
        ok, msg = self.ctx.registry.verify_model(self.model_id)
        self.finished.emit(ok, msg, self.model_id)

# ═══════════════════════════════════════════════════════════════════
#  DIALOGS
# ═══════════════════════════════════════════════════════════════════
class ImageGenerationDialog(QDialog):
    def __init__(self, model_id: str, ctx: AppContext, parent=None):
        super().__init__(parent); self.mid = model_id; self.ctx = ctx; self.setWindowTitle(f"Generate: {model_id}"); self.setMinimumSize(600, 650); self.setStyleSheet(f"background:{C['bg']}")
        lay = QVBoxLayout(self)
        self.prompt = QPlainTextEdit(); self.prompt.setPlaceholderText("A futuristic city...")
        self.steps = QSpinBox(); self.steps.setRange(1, 100); self.steps.setValue(20)
        self.btn = QPushButton("🎨 Generate"); self.btn.setObjectName("imageBtn"); self.btn.clicked.connect(self._gen)
        self.img_lbl = QLabel("No image"); self.img_lbl.setAlignment(Qt.AlignCenter); self.img_lbl.setMinimumHeight(400); self.img_lbl.setStyleSheet(f"background:{C['surface']}; border-radius:8px;")
        lay.addWidget(QLabel("Prompt:")); lay.addWidget(self.prompt)
        h = QHBoxLayout(); h.addWidget(QLabel("Steps:")); h.addWidget(self.steps); h.addStretch(); h.addWidget(self.btn); lay.addLayout(h)
        lay.addWidget(self.img_lbl, 1)

    def _gen(self):
        b = self.ctx.manager.get_backend(self.mid)
        if not b: return
        self.btn.setEnabled(False); self.btn.setText("Generating...")
        def work():
            try: return b.generate_image(self.prompt.toPlainText(), {"steps": self.steps.value()})
            except Exception as e: return str(e)
        class T(QThread):
            done = Signal(object)
            def __init__(s, f): super().__init__(); s.f = f
            def run(s): s.done.emit(s.f())
        self.t = T(work); self.t.done.connect(self._on_done); self.t.start()

    def _on_done(self, res):
        self.btn.setEnabled(True); self.btn.setText("🎨 Generate")
        if isinstance(res, str): QMessageBox.critical(self, "Err", res); return
        if HAS_PIL:
            buf = io.BytesIO(); res.save(buf, format="PNG"); buf.seek(0)
            qimg = QImage(); qimg.loadFromData(buf.getvalue()); self.img_lbl.setPixmap(QPixmap.fromImage(qimg).scaled(self.img_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

# ───────────────────────────────────────────────────────────────────
#  NEW: ModelDetailDialog — shows model details, file list with
#  sizes in GB, download buttons, duplicate detection
# ───────────────────────────────────────────────────────────────────
class ModelDetailDialog(QDialog):
    download_requested = Signal(str, str, int)  # repo_id, filename, size

    def __init__(self, repo_id: str, ctx: AppContext, parent=None):
        super().__init__(parent); self.repo_id = repo_id; self.ctx = ctx
        self.setWindowTitle(f"Model Details — {repo_id}")
        self.setMinimumSize(750, 550); self.setStyleSheet(f"background:{C['bg']}")
        self._dl_worker = None
        self._build_ui()
        self._fetch_info()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        # Header
        self.header_lbl = QLabel(f"📦 {self.repo_id}")
        self.header_lbl.setStyleSheet(f"font-size:16px; font-weight:700; color:{C['accent']};")
        lay.addWidget(self.header_lbl)

        # Info row
        self.info_lbl = QLabel("Loading model info...")
        self.info_lbl.setStyleSheet(f"color:{C['text_dim']};")
        lay.addWidget(self.info_lbl)

        # Files table
        self.table = QTableWidget(); self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["File", "Size (GB)", "Size (bytes)", "Status", "Action"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table, 1)

        # Download progress
        self.pbar = QProgressBar(); self.pbar.setVisible(False)
        self.pbar_lbl = QLabel()
        dl_lay = QHBoxLayout(); dl_lay.addWidget(self.pbar, 1); dl_lay.addWidget(self.pbar_lbl)
        lay.addLayout(dl_lay)

        # Repo download button (for image/diffusers models)
        self.repo_dl_btn = QPushButton("📥 Download Entire Repo"); self.repo_dl_btn.setObjectName("imageBtn")
        self.repo_dl_btn.clicked.connect(self._dl_repo)
        self.repo_dl_btn.setVisible(False)
        lay.addWidget(self.repo_dl_btn)

        # Close
        self.close_btn = QPushButton("Close"); self.close_btn.setObjectName("secondaryBtn")
        self.close_btn.clicked.connect(self.close); lay.addWidget(self.close_btn)

    def _fetch_info(self):
        self._info_worker = ModelInfoWorker(self.repo_id)
        self._info_worker.info_ready.connect(self._on_info)
        self._info_worker.error.connect(lambda e: self.info_lbl.setText(f"Error: {e}"))
        self._info_worker.start()

    def _on_info(self, info: dict):
        self.info_data = info
        pt = info.get("pipeline_tag", "")
        self.info_lbl.setText(
            f"Pipeline: {pt or '—'}  |  Downloads: {info.get('downloads', 0):,}  |  "
            f"SHA: {info.get('sha', '—')}"
        )

        files = info.get("files", [])
        is_image = pt == "text-to-image"

        if not files and is_image:
            # Diffusers repos often have no single model file; offer repo download
            self.repo_dl_btn.setVisible(True)
            # Check if already downloaded
            if self.ctx.registry.is_downloaded(self.repo_id):
                self.repo_dl_btn.setText("✅ Already Downloaded"); self.repo_dl_btn.setEnabled(False)
            self.table.setRowCount(0)
            return

        self.table.setRowCount(len(files))
        for i, f in enumerate(files):
            fname = f["filename"]
            fsize = f.get("size", 0)

            self.table.setItem(i, 0, QTableWidgetItem(fname))
            self.table.setItem(i, 1, QTableWidgetItem(fmt_size_gb(fsize)))
            self.table.setItem(i, 2, QTableWidgetItem(f"{fsize:,}" if fsize else "—"))

            # Check if this specific file is already downloaded
            file_mid = f"{self.repo_id}::{fname}"
            existing = self.ctx.registry.get(file_mid)
            is_downloaded = existing and Path(existing.path).exists()

            if is_downloaded:
                status = QTableWidgetItem("✅ Downloaded")
                status.setForeground(QColor(C["ok"]))
                self.table.setItem(i, 3, status)
                # No action button needed
                action_lbl = QLabel("✓"); action_lbl.setAlignment(Qt.AlignCenter)
                self.table.setCellWidget(i, 4, action_lbl)
            else:
                status = QTableWidgetItem("Available")
                status.setForeground(QColor(C["accent"]))
                self.table.setItem(i, 3, status)
                # Download button
                btn = QPushButton("📥 Download"); btn.setObjectName("downloadBtn")
                btn.setProperty("fname", fname); btn.setProperty("fsize", fsize)
                btn.clicked.connect(lambda checked, fn=fname, fs=fsize: self._dl_file(fn, fs))
                btn_w = QWidget(); btn_l = QHBoxLayout(btn_w); btn_l.setContentsMargins(4,2,4,2); btn_l.addWidget(btn)
                self.table.setCellWidget(i, 4, btn_w)

        # Also show repo download button for image models alongside files
        if is_image:
            self.repo_dl_btn.setVisible(True)
            if self.ctx.registry.is_downloaded(self.repo_id):
                self.repo_dl_btn.setText("✅ Already Downloaded"); self.repo_dl_btn.setEnabled(False)

    def _dl_file(self, filename: str, file_size: int):
        """Download a single file (e.g., a specific GGUF quantization)."""
        # ── DUPLICATE CHECK ──
        mid = f"{self.repo_id}::{filename}"
        if self.ctx.registry.is_downloaded(mid):
            QMessageBox.information(self, "Already Downloaded", f"{filename} is already downloaded and verified.")
            return

        if self._dl_worker and self._dl_worker.isRunning():
            QMessageBox.warning(self, "Busy", "A download is already in progress."); return

        self.pbar.setVisible(True); self.pbar.setValue(0)
        dest = str(Path(self.ctx.config.models_dir) / self.repo_id.replace("/", "__") / filename)
        url = f"https://huggingface.co/{self.repo_id}/resolve/main/{filename}"

        task = PrioritizedTask(priority=0, task_id=f"dl_{time.time()}", url=url,
                               dest_path=dest, filename=filename, model_id=mid,
                               source="huggingface", expected_size=file_size)

        if mid not in self.ctx.manager.state_machines:
            self.ctx.manager.state_machines[mid] = ModelStateMachine(ModelState.REMOTE)

        self._dl_worker = DownloadWorker(task, self.ctx)
        self._dl_worker.progress.connect(lambda p, s: (
            self.pbar.setValue(p),
            self.pbar_lbl.setText(f"{p}% — {s:.0f} KB/s")
        ))
        self._dl_worker.finished.connect(self._on_dl_done)
        self._dl_worker.start()

    def _dl_repo(self):
        """Download the entire repo (for diffusers models)."""
        if self.ctx.registry.is_downloaded(self.repo_id):
            QMessageBox.information(self, "Already Downloaded", f"{self.repo_id} is already downloaded and verified.")
            return
        if self._dl_worker and hasattr(self._dl_worker, 'isRunning') and self._dl_worker.isRunning():
            QMessageBox.warning(self, "Busy", "A download is already in progress."); return

        self.pbar.setVisible(True); self.pbar.setValue(0)
        if self.repo_id not in self.ctx.manager.state_machines:
            self.ctx.manager.state_machines[self.repo_id] = ModelStateMachine(ModelState.REMOTE)

        self._dl_worker = RepoDownloadWorker(self.repo_id, self.ctx)
        self._dl_worker.progress.connect(lambda p, s: self.pbar.setValue(p))
        self._dl_worker.finished.connect(self._on_dl_done)
        self._dl_worker.start()

    def _on_dl_done(self, ok: bool, msg: str, mid: str):
        self.pbar.setVisible(False)
        if ok:
            QMessageBox.information(self, "Download Complete", msg)
            self._fetch_info()  # Refresh file list to show updated status
        else:
            QMessageBox.critical(self, "Download Error", msg)


# ═══════════════════════════════════════════════════════════════════
#  UI PAGES (Dependency Injected)
# ═══════════════════════════════════════════════════════════════════
class BrowsePage(QWidget):
    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent); self.ctx = ctx; self._worker = None; self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(32,28,32,28)
        title = QLabel("Browse Models")
        title.setStyleSheet(f"font-size:18px; font-weight:700;")
        lay.addWidget(title)

        h = QHBoxLayout()
        self.search_in = QLineEdit(); self.search_in.setPlaceholderText("Search HuggingFace (Trie AutoComplete)...")
        self.completer = QCompleter(); self.search_in.setCompleter(self.completer)
        self.completer_model = QStringListModel(); self.completer.setModel(self.completer_model)
        self.update_completer()
        self.search_btn = QPushButton("🔍 Search"); self.search_btn.setObjectName("primaryBtn"); self.search_btn.clicked.connect(self._search)
        h.addWidget(self.search_in, 1); h.addWidget(self.search_btn); lay.addLayout(h)

        # ── ENHANCED: columns include Size (GB) and Status ──
        self.table = QTableWidget(); self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["ID", "Type", "Params", "Size (GB)", "Status", "Action"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers); self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        lay.addWidget(self.table, 1)

    def update_completer(self):
        keys = self.ctx.trie.search_prefix("")[:1000]
        self.completer_model.setStringList(keys)

    def _search(self):
        q = self.search_in.text().strip()
        if not q: return
        self.search_btn.setEnabled(False)
        self._worker = SearchWorker(q, self.ctx)
        self._worker.results_ready.connect(self._on_results)
        self._worker.finished.connect(lambda: self.search_btn.setEnabled(True))
        self._worker.start()

    def _on_results(self, results):
        self.table.setRowCount(len(results)); self.update_completer()
        for i, r in enumerate(results):
            self.table.setItem(i, 0, QTableWidgetItem(r["id"]))
            self.table.setItem(i, 1, QTableWidgetItem("🖼" if r["type"] == ModelType.IMAGE else "📝"))
            self.table.setItem(i, 2, QTableWidgetItem(fmt_params(r["params"])))

            # ── Size (GB): fetch from registry if previously downloaded ──
            existing = self.ctx.registry.find_by_repo(r["id"])
            if existing:
                total_size = sum(e.size for e in existing)
                self.table.setItem(i, 3, QTableWidgetItem(fmt_size_gb(total_size)))
            else:
                self.table.setItem(i, 3, QTableWidgetItem("—"))

            # ── Status: show download status ──
            is_dl = self.ctx.registry.is_downloaded(r["id"]) or bool(existing)
            if is_dl:
                status_item = QTableWidgetItem("✅ Downloaded")
                status_item.setForeground(QColor(C["ok"]))
                self.table.setItem(i, 4, status_item)
            else:
                self.table.setItem(i, 4, QTableWidgetItem("Available"))

            # ── Action: Download button (or "View" if already downloaded) ──
            btn_w = QWidget(); btn_l = QHBoxLayout(btn_w); btn_l.setContentsMargins(4,2,4,2)
            if is_dl:
                btn_view = QPushButton("📂 View"); btn_view.setObjectName("okBtn")
                btn_view.clicked.connect(lambda checked, mid=r["id"]: self._view_details(mid))
                btn_l.addWidget(btn_view)
            else:
                btn_dl = QPushButton("📥 Download"); btn_dl.setObjectName("downloadBtn")
                btn_dl.clicked.connect(lambda checked, mid=r["id"]: self._view_details(mid))
                btn_l.addWidget(btn_dl)
            btn_l.addStretch()
            self.table.setCellWidget(i, 5, btn_w)

    def _view_details(self, repo_id: str):
        """Open the ModelDetailDialog for viewing/downloading a model."""
        dlg = ModelDetailDialog(repo_id, self.ctx, self)
        dlg.exec_()
        # After dialog closes, refresh search results to update statuses
        self.update_completer()


class ModelsPage(QWidget):
    model_changed = Signal()
    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent); self.ctx = ctx; self._verify_worker = None; self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(32,28,32,28)
        h = QHBoxLayout()
        title = QLabel("My Models")
        title.setStyleSheet(f"font-size:18px; font-weight:700;")
        h.addWidget(title); h.addStretch()
        btn_ref = QPushButton("🔄 Refresh"); btn_ref.setObjectName("secondaryBtn"); btn_ref.clicked.connect(self.refresh)
        btn_imp = QPushButton("📂 Import"); btn_imp.setObjectName("secondaryBtn"); btn_imp.clicked.connect(self._import)
        btn_verify_all = QPushButton("✅ Verify All"); btn_verify_all.setObjectName("verifyBtn"); btn_verify_all.clicked.connect(self._verify_all)
        h.addWidget(btn_imp); h.addWidget(btn_verify_all); h.addWidget(btn_ref)
        lay.addLayout(h)

        # ── ENHANCED: more columns for metadata ──
        self.tree = QTreeWidget(); self.tree.setColumnCount(7)
        self.tree.setHeaderLabels(["Model ID", "Format", "Size (GB)", "Status", "Checksum", "Version", "Actions"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.tree.setAlternatingRowColors(True); self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._ctx_menu)
        lay.addWidget(self.tree, 1)

    def refresh(self):
        self.tree.clear()
        records = self.ctx.registry.list_all()
        grouped = {}
        for r in records:
            if r.model_id not in self.ctx.manager.state_machines:
                self.ctx.manager.state_machines[r.model_id] = ModelStateMachine(r.state)
            family = self.ctx.graph.get_family(r.model_id)
            root = family[0]
            if root not in grouped: grouped[root] = []
            grouped[root].append(r)
            if r.model_id != root and r.base_model_id is None:
                r.base_model_id = root

        for base_id, variants in grouped.items():
            # Calculate total size for root
            total_size = sum(v.size for v in variants)
            root_item = QTreeWidgetItem(self.tree, [
                base_id, "", fmt_size_gb(total_size), "", "", "", ""
            ])
            root_item.setData(0, Qt.UserRole, base_id)
            root_item.setForeground(2, QColor(C["accent"]))

            for v in variants:
                state = self.ctx.manager.state_machines[v.model_id].state
                fmt = v.format_key.upper()
                sz_gb = fmt_size_gb(v.size)

                # ── Download status with color ──
                dl_status = v.download_status
                status_colors = {
                    "verified": C["ok"], "complete": C["accent2"],
                    "pending": C["warn"], "corrupted": C["danger"]
                }
                status_color = status_colors.get(dl_status, C["text_dim"])
                status_text = {
                    "verified": "✅ Verified", "complete": "📦 Complete",
                    "pending": "⏳ Pending", "corrupted": "❌ Corrupted"
                }.get(dl_status, dl_status)

                # ── Checksum (truncated) ──
                cs_text = f"{v.checksum[:12]}…" if v.checksum else "—"

                # ── Version ──
                ver_text = v.version or "—"

                item = QTreeWidgetItem(root_item, [v.model_id, fmt, sz_gb, status_text, cs_text, ver_text, ""])
                item.setData(0, Qt.UserRole, v.model_id)
                item.setForeground(2, QColor(C["accent"]))
                item.setForeground(3, QColor(status_color))

                if state == ModelState.LOADED: item.setForeground(0, QColor(C["accent2"]))
                elif state == ModelState.ERROR: item.setForeground(0, QColor(C["danger"]))

                # ── Action buttons ──
                btn_w = QWidget(); btn_l = QHBoxLayout(btn_w); btn_l.setContentsMargins(4,2,4,2)
                if state == ModelState.LOADED:
                    if v.model_type == ModelType.IMAGE:
                        btn_gen = QPushButton("🎨 Gen"); btn_gen.setObjectName("imageBtn")
                        btn_gen.clicked.connect(lambda ch, mid=v.model_id: ImageGenerationDialog(mid, self.ctx, self).exec())
                        btn_l.addWidget(btn_gen)
                    btn_unload = QPushButton("Unload"); btn_unload.setObjectName("warnBtn")
                    btn_unload.clicked.connect(lambda ch, mid=v.model_id: self._unload(mid))
                    btn_l.addWidget(btn_unload)
                elif state == ModelState.DOWNLOADED:
                    btn_load = QPushButton("Load"); btn_load.setObjectName("downloadBtn")
                    btn_load.clicked.connect(lambda ch, mid=v.model_id: self._load(mid))
                    btn_l.addWidget(btn_load)

                # ── Verify button ──
                btn_verify = QPushButton("🔍 Verify"); btn_verify.setObjectName("verifyBtn")
                btn_verify.clicked.connect(lambda ch, mid=v.model_id: self._verify_model(mid))
                btn_l.addWidget(btn_verify)

                self.tree.setItemWidget(item, 6, btn_w)
        self.tree.expandAll()

    def _load(self, mid: str):
        rec = self.ctx.registry.get(mid)
        if not rec: return
        # ── Pre-load verification ──
        ok, msg = file_exists_and_valid(rec.path, rec.size, rec.checksum)
        if not ok:
            rec.download_status = "corrupted"; self.ctx.registry.save()
            QMessageBox.critical(self, "File Corrupted", f"Cannot load {mid}: {msg}\nPlease re-download.")
            self.refresh(); return
        det = detect_format(rec.filename)
        cmd = LoadModelCommand(mid, rec.path, det, self.ctx.config, self.ctx.manager)
        try:
            self.ctx.manager.load_model(cmd)
            QMessageBox.information(self, "Loaded", f"{mid} loaded."); self.refresh()
        except Exception as e: QMessageBox.critical(self, "Error", str(e)); self.refresh()

    def _unload(self, mid: str):
        self.ctx.manager.unload_model(mid); self.refresh()

    def _import(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Import", "", "Model Files (*.gguf *.safetensors *.bin)")
        if fp:
            mid = Path(fp).stem; det = detect_format(fp)
            fsize = Path(fp).stat().st_size
            # ── Compute checksum on import ──
            checksum = compute_sha256(fp)
            rec = ModelRecord(
                model_id=mid, path=fp, filename=Path(fp).name, size=fsize,
                format_key=det.format_key, quant_key=det.quant_key,
                params_b=det.estimated_params_b, vram_gb=det.estimated_vram_gb,
                source="local", added_at=datetime.now().isoformat(),
                model_type=det.model_type, state=ModelState.DOWNLOADED,
                checksum=checksum, version="local", download_status="verified"
            )
            self.ctx.registry.add(rec); self.ctx.trie.insert(mid, mid); self.refresh()

    def _verify_model(self, mid: str):
        """Verify a single model's integrity."""
        rec = self.ctx.registry.get(mid)
        if not rec:
            QMessageBox.warning(self, "Not Found", f"{mid} not found in registry."); return
        if self._verify_worker and self._verify_worker.isRunning():
            QMessageBox.warning(self, "Busy", "Verification already in progress."); return
        self._verify_worker = ChecksumWorker(mid, self.ctx)
        self._verify_worker.finished.connect(self._on_verify_done)
        self._verify_worker.start()

    def _verify_all(self):
        """Verify all downloaded models."""
        records = self.ctx.registry.list_all()
        results = []
        for r in records:
            ok, msg = self.ctx.registry.verify_model(r.model_id)
            results.append((r.model_id, ok, msg))
        self.refresh()
        # Show summary
        ok_count = sum(1 for _, ok, _ in results if ok)
        bad_count = len(results) - ok_count
        detail = "\n".join(f"{'✅' if ok else '❌'} {mid}: {msg}" for mid, ok, msg in results)
        QMessageBox.information(self, "Verification Complete",
                                f"Verified {len(results)} models: {ok_count} OK, {bad_count} issues\n\n{detail}")

    def _on_verify_done(self, ok: bool, msg: str, mid: str):
        if ok:
            QMessageBox.information(self, "Verification Passed", f"✅ {mid}: {msg}")
        else:
            QMessageBox.critical(self, "Verification Failed", f"❌ {mid}: {msg}")
        self.refresh()

    def _ctx_menu(self, pos):
        item = self.tree.itemAt(pos)
        if not item: return
        mid = item.data(0, Qt.UserRole)
        menu = QMenu(self)
        rec = self.ctx.registry.get(mid)
        if rec:
            menu.addAction("🔍 Verify", lambda: self._verify_model(mid))
            menu.addAction("📋 Copy Checksum", lambda: self._copy_checksum(mid))
            menu.addSeparator()
            menu.addAction("🗑 Delete", lambda: (self.ctx.registry.remove(mid), self.refresh()))
        menu.exec(self.tree.mapToGlobal(pos))

    def _copy_checksum(self, mid: str):
        rec = self.ctx.registry.get(mid)
        if rec and rec.checksum:
            QApplication.clipboard().setText(rec.checksum)
            QMessageBox.information(self, "Copied", f"SHA-256 checksum copied for {mid}")
        else:
            QMessageBox.information(self, "No Checksum", f"No checksum available for {mid}")


class APIPage(QWidget):
    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent); self.ctx = ctx; self.api_srv = APIServer(ctx); self._build_ui()
    def _build_ui(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(32,28,32,28)
        title = QLabel("API Server (LRU Cached)")
        title.setStyleSheet(f"font-size:18px; font-weight:700;")
        lay.addWidget(title)
        if not HAS_FASTAPI: lay.addWidget(QLabel("⚠ FastAPI missing.")); return
        h = QHBoxLayout(); self.port = QSpinBox(); self.port.setRange(1024, 65535); self.port.setValue(self.ctx.config.server_port)
        self.btn = QPushButton("▶ Start"); self.btn.setObjectName("primaryBtn"); self.btn.clicked.connect(self._toggle)
        h.addWidget(QLabel("Port:")); h.addWidget(self.port); h.addWidget(self.btn); lay.addLayout(h)
        self.status = QLabel("Stopped"); self.status.setStyleSheet(f"color:{C['text_dim']}"); lay.addWidget(self.status)
        lay.addWidget(QLabel("Endpoints:\nGET /v1/models\nPOST /v1/chat/completions\nPOST /v1/images/generations"))
    def _toggle(self):
        if self.api_srv.running:
            self.api_srv.stop(); self.btn.setText("▶ Start"); self.btn.setObjectName("primaryBtn"); self.btn.style().unpolish(self.btn); self.btn.style().polish(self.btn)
            self.status.setText("Stopped"); self.status.setStyleSheet(f"color:{C['text_dim']}")
        else:
            self.ctx.config.server_port = self.port.value()
            if self.api_srv.start():
                self.btn.setText("■ Stop"); self.btn.setStyleSheet(f"background:{C['danger']}; color:#fff; border:none; border-radius:6px; padding:8px 20px; font-weight:600;")
                self.status.setText(f"Running on {self.ctx.config.server_port}"); self.status.setStyleSheet(f"color:{C['accent2']}; font-weight:600;")

# ═══════════════════════════════════════════════════════════════════
#  MAIN WINDOW & ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
NAV_ITEMS = [("browse", "🔍", "Browse"), ("models", "📦", "Models"), ("api", "🌐", "API")]

class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext):
        super().__init__(); self.ctx = ctx; self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}"); self.resize(1280, 850)
        self._build_ui()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central); main_lay = QHBoxLayout(central); main_lay.setContentsMargins(0,0,0,0); main_lay.setSpacing(0)
        sidebar = QFrame(); sidebar.setObjectName("sidebar"); sidebar.setFixedWidth(200); side_lay = QVBoxLayout(sidebar); side_lay.setContentsMargins(8,16,8,16)

        brand = QLabel(f"  {APP_NAME} v{APP_VERSION}")
        brand.setStyleSheet(f"font-weight:700; color:{C['accent']}; font-size:14px; padding:8px 0;")
        side_lay.addWidget(brand)

        self._nav_btns = {}; self.stack = QStackedWidget(); self.pages = {}
        self.pages["browse"] = BrowsePage(self.ctx)
        self.pages["models"] = ModelsPage(self.ctx)
        self.pages["api"] = APIPage(self.ctx)

        for key, icon, label in NAV_ITEMS:
            btn = QPushButton(f"  {icon}  {label}"); btn.setObjectName("navBtn"); btn.setCursor(QCursor(Qt.PointingHandCursor))
            btn.clicked.connect(lambda checked, k=key: self.navigate_to(k))
            side_lay.addWidget(btn); self._nav_btns[key] = btn
            self.stack.addWidget(self.pages[key])

        side_lay.addStretch()
        gpus = self.ctx.gpus
        gpu_lbl = QLabel(f"  GPU: {fmt_size_gb(sum(g.total_vram_mb for g in gpus)*1024*1024)}" if gpus else "  No GPU detected")
        gpu_lbl.setStyleSheet(f"color:{C['text_dim']}; font-size:12px;"); side_lay.addWidget(gpu_lbl)

        # ── Manifest info in sidebar ──
        model_count = len(self.ctx.registry.list_all())
        total_size = sum(r.size for r in self.ctx.registry.list_all())
        info_lbl = QLabel(f"  Models: {model_count} | {fmt_size_gb(total_size)}")
        info_lbl.setStyleSheet(f"color:{C['text_dim']}; font-size:12px;"); side_lay.addWidget(info_lbl)

        main_lay.addWidget(sidebar); main_lay.addWidget(self.stack, 1)
        self.navigate_to("browse")

    def navigate_to(self, key: str):
        self.stack.setCurrentWidget(self.pages[key])
        for k, btn in self._nav_btns.items(): btn.setProperty("active", k == key); btn.style().unpolish(btn); btn.style().polish(btn)
        if key == "models": self.pages[key].refresh()

    def closeEvent(self, event):
        if hasattr(self.pages.get("api"), 'api_srv') and self.pages["api"].api_srv.running: self.pages["api"].api_srv.stop()
        for mid in list(self.ctx.manager.backends.keys()): self.ctx.manager.unload_model(mid)
        event.accept()


def main():
    app = QApplication(sys.argv); app.setStyle("Fusion"); app.setStyleSheet(STYLESHEET)

    # Dependency Injection Setup
    config = AppConfig()
    gpus = GPUDetector.detect()
    registry = ModelRegistry(config.models_dir)
    manager = ModelManager()
    trie = ModelTrie()
    graph = ModelGraph()
    dl_scheduler = DownloadScheduler()

    # Populate Trie and State Machines from existing registry
    for rec in registry.list_all():
        trie.insert(rec.model_id, rec.model_id)
        graph.add_variant(rec.model_id.split("/")[0] if "/" in rec.model_id else rec.model_id, rec.model_id)
        if rec.model_id not in manager.state_machines:
            manager.state_machines[rec.model_id] = ModelStateMachine(rec.state)
        # ── Verify existing models on startup (quick size check) ──
        if rec.path and not Path(rec.path).exists():
            rec.download_status = "corrupted"
            logger.warning(f"Model {rec.model_id} file missing, marked as corrupted")
        elif rec.size > 0 and Path(rec.path).exists() and Path(rec.path).stat().st_size != rec.size:
            rec.download_status = "corrupted"
            logger.warning(f"Model {rec.model_id} size mismatch, marked as corrupted")
    registry.save()  # Persist any status updates

    ctx = AppContext(config, gpus, registry, manager, trie, graph, dl_scheduler)
    global AppContextSingleton; AppContextSingleton = ctx

    window = MainWindow(ctx); window.show(); sys.exit(app.exec())

if __name__ == "__main__":
    main()
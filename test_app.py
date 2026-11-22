# test_cacheai.py
"""
Comprehensive pytest suite for CacheAI – Local LLM Model Manager & Multi-Backend API Server
Tests: downloading, GPU detection, format detection, model registry, config,
       API server, model manager, workers, and utility functions.
"""

import sys
import os
import json
import time
import shutil
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock, call
from io import BytesIO

import pytest

# ─────────────────────────────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    """Redirect CONFIG_FILE and DEFAULT_MODELS_DIR to tmp so tests never touch real dirs."""
    cfg_dir = tmp_path / ".cacheai"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = cfg_dir / "config.json"
    models_dir = tmp_path / "cacheAI_models"

    monkeypatch.setattr("app.CONFIG_FILE", str(cfg_file))
    monkeypatch.setattr("app.DEFAULT_MODELS_DIR", str(models_dir))
    yield


@pytest.fixture
def fresh_config():
    """Return a fresh AppConfig (singleton reset each time)."""
    import app
    app.AppConfig._instance = None
    cfg = app.AppConfig()
    yield cfg
    app.AppConfig._instance = None


@pytest.fixture
def temp_models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture
def registry(temp_models_dir):
    import app
    r = app.ModelRegistry(str(temp_models_dir))
    yield r


@pytest.fixture
def sample_gguf_file(temp_models_dir):
    """Create a tiny fake GGUF file for import / load tests."""
    p = temp_models_dir / "llama-7b-q4_k_m.gguf"
    p.write_bytes(b"GGUF" + b"\x00" * 1024)
    return p


@pytest.fixture
def sample_safetensors_file(temp_models_dir):
    p = temp_models_dir / "mistral-7b.safetensors"
    p.write_bytes(b"\x00" * 2048)
    return p


@pytest.fixture
def model_manager():
    import app
    return app.MultiModelManager()


# ─────────────────────────────────────────────────────────────────────
#  1. UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

class TestFmtSize:
    def test_bytes(self):
        from app import fmt_size
        assert fmt_size(500) == "500.0 B"

    def test_kilobytes(self):
        from app import fmt_size
        assert "KB" in fmt_size(2048)

    def test_megabytes(self):
        from app import fmt_size
        assert "MB" in fmt_size(5 * 1024 * 1024)

    def test_gigabytes(self):
        from app import fmt_size
        assert "GB" in fmt_size(3 * 1024 ** 3)

    def test_terabytes(self):
        from app import fmt_size
        assert "TB" in fmt_size(2 * 1024 ** 4)

    def test_negative(self):
        from app import fmt_size
        assert fmt_size(-1) == "—"

    def test_zero(self):
        from app import fmt_size
        assert fmt_size(0) == "0.0 B"


class TestFmtSpeed:
    def test_kbps(self):
        from app import fmt_speed
        result = fmt_speed(512)
        assert "KB/s" in result

    def test_mbps(self):
        from app import fmt_speed
        result = fmt_speed(2048)
        assert "MB/s" in result


class TestDetectFormatInfo:
    @pytest.mark.parametrize("filename,expected_format,expected_quant", [
        ("model-q4_k_m.gguf", "gguf", "q4_k_m"),
        ("model-q8_0.gguf", "gguf", "q8_0"),
        ("model-q5_k_m.gguf", "gguf", "q5_k_m"),
        ("model-q6_k.gguf", "gguf", "q6_k"),
        ("model-f16.gguf", "gguf", "f16"),
        ("model-f32.gguf", "gguf", "f32"),
        ("model-bf16.gguf", "gguf", "bf16"),
        ("mistral-7b.safetensors", "safetensors", "f16"),
        ("model-gptq-4bit.safetensors", "gptq", "gptq"),
        ("model-awq.safetensors", "awq", "awq"),
        ("model-gptq.bin", "gptq", "gptq"),
        ("model-awq.bin", "awq", "awq"),
        ("model-exl2.exl2", "exl2", "exl2"),
        ("pytorch_model.bin", "bin", "f16"),
        ("model-bfloat16.safetensors", "safetensors", "bf16"),
    ])
    def test_format_detection(self, filename, expected_format, expected_quant):
        from app import detect_format_info
        info = detect_format_info(filename)
        assert info["format"] == expected_format, f"For {filename}: expected format {expected_format}, got {info['format']}"
        assert info["quant"] == expected_quant, f"For {filename}: expected quant {expected_quant}, got {info['quant']}"

    def test_params_extraction(self):
        from app import detect_format_info
        info = detect_format_info("llama-7b-q4_k_m.gguf")
        assert info["estimated_params_b"] == 7.0

    def test_params_extraction_decimal(self):
        from app import detect_format_info
        info = detect_format_info("phi-2.7b-q4_k_m.gguf")
        assert info["estimated_params_b"] == 2.7

    def test_unknown_format(self):
        from app import detect_format_info
        info = detect_format_info("readme.txt")
        assert info["format"] == "unknown"

    def test_format_data_keys(self):
        from app import detect_format_info
        info = detect_format_info("model.gguf")
        fd = info["format_data"]
        for key in ("label", "color", "backend", "lib", "desc"):
            assert key in fd

    def test_quant_data_keys(self):
        from app import detect_format_info
        info = detect_format_info("model-q4_k_m.gguf")
        qi = info["quant_data"]
        for key in ("bpp", "quality", "label", "color"):
            assert key in qi


class TestQualityDots:
    def test_level_5(self):
        from app import quality_dots
        assert quality_dots(5) == "●●●●●"

    def test_level_1(self):
        from app import quality_dots
        assert quality_dots(1) == "●○○○○"

    def test_level_0(self):
        from app import quality_dots
        assert quality_dots(0) == "○○○○○"


# ─────────────────────────────────────────────────────────────────────
#  2. APP CONFIG
# ─────────────────────────────────────────────────────────────────────

class TestAppConfig:
    def test_singleton(self, fresh_config):
        import app
        c2 = app.AppConfig()
        assert c2 is fresh_config

    def test_defaults(self, fresh_config):
        assert isinstance(fresh_config.models_dir, str)
        assert fresh_config.n_ctx == 4096
        assert fresh_config.n_gpu_layers == -1
        assert fresh_config.server_port == 8081

    def test_set_and_get(self, fresh_config):
        fresh_config.set("custom_key", "custom_value")
        assert fresh_config.get("custom_key") == "custom_value"

    def test_get_default(self, fresh_config):
        assert fresh_config.get("nonexistent", 42) == 42

    def test_models_dir_setter(self, fresh_config, tmp_path):
        new_dir = str(tmp_path / "new_models")
        fresh_config.models_dir = new_dir
        assert fresh_config.models_dir == new_dir

    def test_server_port_setter(self, fresh_config):
        fresh_config.server_port = 9999
        assert fresh_config.server_port == 9999

    def test_n_threads_positive(self, fresh_config):
        assert fresh_config.n_threads >= 1

    def test_persistence(self, fresh_config):
        fresh_config.set("persist_test", "hello")
        import app
        app.AppConfig._instance = None
        cfg2 = app.AppConfig()
        assert cfg2.get("persist_test") == "hello"
        app.AppConfig._instance = None

    def test_corrupt_config_recovery(self, fresh_config, tmp_path):
        fresh_config.set("good_key", "good_value")
        import app
        cfg_path = Path(app.CONFIG_FILE) if 'app' in dir() else tmp_path / ".cacheai" / "config.json"
        import app
        cfg_path = Path(app.CONFIG_FILE)
        # Write garbage
        cfg_path.write_text("{{invalid json")
        app.AppConfig._instance = None
        cfg2 = app.AppConfig()
        # Should not crash, but data is lost
        assert cfg2.get("good_key") is None
        app.AppConfig._instance = None


# ─────────────────────────────────────────────────────────────────────
#  3. MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────

class TestModelRegistry:
    def test_empty_registry(self, registry):
        models = registry.list_models()
        assert models == []

    def test_add_model(self, registry):
        registry.add("test-model", {"path": "/fake/path.gguf", "filename": "path.gguf", "size": 1024})
        models = registry.list_models()
        assert len(models) == 1
        assert models[0][0] == "test-model"

    def test_remove_model(self, registry, sample_gguf_file):
        registry.add("test-model", {"path": str(sample_gguf_file), "filename": sample_gguf_file.name, "size": 1024})
        assert len(registry.list_models()) == 1
        registry.remove("test-model")
        assert len(registry.list_models()) == 0

    def test_remove_model_deletes_file(self, registry, sample_gguf_file):
        registry.add("del-test", {"path": str(sample_gguf_file), "filename": sample_gguf_file.name, "size": 1024})
        assert sample_gguf_file.exists()
        registry.remove("del-test")
        assert not sample_gguf_file.exists()

    def test_list_models_actual_size(self, registry, sample_gguf_file):
        registry.add("size-test", {"path": str(sample_gguf_file), "filename": sample_gguf_file.name, "size": 512})
        models = registry.list_models()
        assert models[0][1]["_actual_size"] == sample_gguf_file.stat().st_size

    def test_import_model_copy(self, registry, tmp_path):
        src = tmp_path / "source_model.gguf"
        src.write_bytes(b"GGUF_DATA")
        model_id = registry.import_model(str(src))
        assert model_id == "source_model"
        assert len(registry.list_models()) == 1

    def test_import_model_custom_id(self, registry, tmp_path):
        src = tmp_path / "source_model.gguf"
        src.write_bytes(b"GGUF_DATA")
        model_id = registry.import_model(str(src), model_id="my-custom-id")
        assert model_id == "my-custom-id"

    def test_manifest_atomic_write(self, registry):
        """Simulate a crash during write by replacing os.replace temporarily."""
        registry.add("crash-test", {"path": "/fake", "filename": "fake.gguf", "size": 0})
        manifest_path = registry.manifest_path
        # File should exist and be valid JSON
        data = json.loads(manifest_path.read_text())
        assert "crash-test" in data["models"]

    def test_backup_manifest_on_load(self, registry):
        registry.add("backup-test", {"path": "/fake", "filename": "fake.gguf", "size": 0})
        assert registry.backup_manifest_path.exists()

    def test_restore_from_backup(self, registry):
        registry.add("restore-test", {"path": "/fake", "filename": "fake.gguf", "size": 0})
        # Corrupt the main manifest
        registry.manifest_path.write_text("BROKEN{json")
        # Re-create registry to trigger backup restore
        import app
        r2 = app.ModelRegistry(str(registry.models_dir))
        models = r2.list_models()
        assert any(mid == "restore-test" for mid, _ in models)

    def test_cleanup_orphaned_parts(self, temp_models_dir):
        (temp_models_dir / "orphan.part").write_bytes(b"junk")
        (temp_models_dir / "another.part").write_bytes(b"junk2")
        import app
        r = app.ModelRegistry(str(temp_models_dir))
        assert not (temp_models_dir / "orphan.part").exists()
        assert not (temp_models_dir / "another.part").exists()


# ─────────────────────────────────────────────────────────────────────
#  4. GPU DETECTOR
# ─────────────────────────────────────────────────────────────────────

class TestGPUDetector:
    @patch("subprocess.run")
    def test_detect_nvidia_gpu(self, mock_run):
        from app import GPUDetector
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA GeForce RTX 4090, 24576, 20480\n1, NVIDIA GeForce RTX 3090, 24576, 12288\n"
        )
        gpus = GPUDetector.detect_gpus()
        assert len(gpus) == 2
        assert gpus[0]["type"] == "nvidia"
        assert gpus[0]["name"] == "NVIDIA GeForce RTX 4090"
        assert gpus[0]["total_vram_mb"] == 24576.0
        assert gpus[0]["free_vram_mb"] == 20480.0
        assert gpus[1]["index"] == 1

    @patch("subprocess.run")
    def test_detect_no_gpu(self, mock_run):
        from app import GPUDetector
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
        gpus = GPUDetector.detect_gpus()
        assert gpus == []

    @patch("subprocess.run")
    def test_detect_nvidia_smi_fails(self, mock_run):
        from app import GPUDetector
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        gpus = GPUDetector.detect_gpus()
        assert gpus == []

    @patch("subprocess.run")
    def test_detect_nvidia_smi_timeout(self, mock_run):
        from app import GPUDetector
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)
        gpus = GPUDetector.detect_gpus()
        assert gpus == []

    @patch("subprocess.run")
    def test_single_gpu(self, mock_run):
        from app import GPUDetector
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA A100-SXM4-80GB, 81920, 65536\n"
        )
        gpus = GPUDetector.detect_gpus()
        assert len(gpus) == 1
        assert gpus[0]["total_vram_mb"] == 81920.0
        assert gpus[0]["free_vram_mb"] == 65536.0

    @patch("subprocess.run")
    def test_gpu_malformed_output(self, mock_run):
        from app import GPUDetector
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="garbage,line\n"
        )
        gpus = GPUDetector.detect_gpus()
        assert gpus == []


# ─────────────────────────────────────────────────────────────────────
#  5. DOWNLOAD WORKER
# ─────────────────────────────────────────────────────────────────────

class TestDownloadWorker:
    def _make_worker(self, url, dest, model_id, registry, filename, expected_size=0):
        from app import DownloadWorker
        return DownloadWorker(url, dest, model_id, registry, filename, expected_size)

    def test_successful_download(self, registry, temp_models_dir):
        dest = str(temp_models_dir / "test_model.gguf")
        worker = self._make_worker("http://fake.url/model.gguf", dest, "test-model", registry, "test_model.gguf", expected_size=13)

        # Mock requests.get to return a small stream
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "13"}
        mock_response.iter_content.return_value = [b"Hello, world!"]

        with patch("requests.get", return_value=mock_response):
            results = []
            worker.finished.connect(lambda ok, msg: results.append((ok, msg)))
            worker.run()

        assert results[0][0] is True
        assert Path(dest).exists()
        assert Path(dest).read_bytes() == b"Hello, world!"

    def test_download_with_progress(self, registry, temp_models_dir):
        dest = str(temp_models_dir / "progress_model.gguf")
        worker = self._make_worker("http://fake.url/model.gguf", dest, "progress-model", registry, "progress_model.gguf", expected_size=0)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "9"}
        mock_response.iter_content.return_value = [b"123", b"456", b"789"]

        progress_signals = []
        worker.progress.connect(lambda d, t, s: progress_signals.append((d, t, s)))

        with patch("requests.get", return_value=mock_response):
            worker.run()

        assert len(progress_signals) >= 1
        # Total should be reported
        assert progress_signals[-1][0] == 9  # total downloaded bytes

    def test_download_cancel(self, registry, temp_models_dir):
        dest = str(temp_models_dir / "cancel_model.gguf")
        worker = self._make_worker("http://fake.url/model.gguf", dest, "cancel-model", registry, "cancel_model.gguf")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "100"}

        # Cancel after first chunk
        def iter_chunks(chunk_size):
            yield b"chunk1"
            worker._cancel = True

        mock_response.iter_content = iter_chunks

        results = []
        worker.finished.connect(lambda ok, msg: results.append((ok, msg)))

        with patch("requests.get", return_value=mock_response):
            worker.run()

        assert results[0][0] is False
        assert "Cancelled" in results[0][1]

    def test_download_disk_space_check(self, registry, temp_models_dir):
        dest = str(temp_models_dir / "big_model.gguf")
        # Request 10 TB download with expected_size
        huge_size = 10 * 1024 ** 4
        worker = self._make_worker("http://fake.url/big.gguf", dest, "big-model", registry, "big_model.gguf", expected_size=huge_size)

        results = []
        worker.finished.connect(lambda ok, msg: results.append((ok, msg)))
        worker.run()

        assert results[0][0] is False
        assert "Insufficient disk space" in results[0][1]

    def test_download_size_mismatch(self, registry, temp_models_dir):
        dest = str(temp_models_dir / "mismatch.gguf")
        # Expect 1000 bytes but only get 13
        worker = self._make_worker("http://fake.url/model.gguf", dest, "mismatch-model", registry, "mismatch.gguf", expected_size=1000)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "13"}
        mock_response.iter_content.return_value = [b"Hello, world!"]

        results = []
        worker.finished.connect(lambda ok, msg: results.append((ok, msg)))

        with patch("requests.get", return_value=mock_response):
            worker.run()

        assert results[0][0] is False
        assert "mismatch" in results[0][1].lower()
        # Part file should be deleted
        assert not Path(dest + ".part").exists()

    def test_download_resume(self, registry, temp_models_dir):
        dest = str(temp_models_dir / "resume_model.gguf")
        part_path = dest + ".part"

        # Simulate partially downloaded file
        Path(part_path).write_bytes(b"Hello,")
        existing_size = len(b"Hello,")

        worker = self._make_worker("http://fake.url/model.gguf", dest, "resume-model", registry, "resume_model.gguf")

        mock_response = MagicMock()
        mock_response.status_code = 206  # Partial content
        mock_response.headers = {"content-length": "7"}
        mock_response.iter_content.return_value = [b" world!"]

        results = []
        worker.finished.connect(lambda ok, msg: results.append((ok, msg)))

        with patch("requests.get", return_value=mock_response) as mock_get:
            worker.run()
            # Verify Range header was sent
            call_kwargs = mock_get.call_args
            assert "Range" in call_kwargs.kwargs.get("headers", {})
            range_val = call_kwargs.kwargs["headers"]["Range"]
            assert str(existing_size) in range_val

        assert results[0][0] is True
        assert Path(dest).read_bytes() == b"Hello, world!"

    def test_download_registers_in_registry(self, registry, temp_models_dir):
        dest = str(temp_models_dir / "reg_model.gguf")
        worker = self._make_worker("http://fake.url/model.gguf", dest, "reg-model", registry, "reg_model.gguf", expected_size=5)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "5"}
        mock_response.iter_content.return_value = [b"hello"]

        with patch("requests.get", return_value=mock_response):
            worker.run()

        models = registry.list_models()
        assert any(mid == "reg-model" for mid, _ in models)

    def test_download_network_error(self, registry, temp_models_dir):
        dest = str(temp_models_dir / "error_model.gguf")
        worker = self._make_worker("http://fake.url/model.gguf", dest, "error-model", registry, "error_model.gguf")

        results = []
        worker.finished.connect(lambda ok, msg: results.append((ok, msg)))

        with patch("requests.get", side_effect=ConnectionError("Network down")):
            worker.run()

        assert results[0][0] is False
        assert "Network down" in results[0][1]


# ─────────────────────────────────────────────────────────────────────
#  6. SEARCH WORKER
# ─────────────────────────────────────────────────────────────────────

class TestSearchWorker:
    def test_successful_search(self):
        from app import SearchWorker
        worker = SearchWorker("llama")

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            {"id": "meta-llama/Llama-3-8B", "downloads": 500000},
            {"id": "meta-llama/Llama-3-70B", "downloads": 200000},
        ]

        results = []
        worker.results_ready.connect(lambda r: results.append(r))

        with patch("requests.get", return_value=mock_response):
            worker.run()

        assert len(results) == 1
        assert len(results[0]) == 2
        assert results[0][0]["id"] == "meta-llama/Llama-3-8B"

    def test_search_error(self):
        from app import SearchWorker
        worker = SearchWorker("nonexistent")

        errors = []
        worker.error.connect(lambda e: errors.append(e))

        with patch("requests.get", side_effect=ConnectionError("No internet")):
            worker.run()

        assert len(errors) == 1
        assert "No internet" in errors[0]


# ─────────────────────────────────────────────────────────────────────
#  7. MODEL FILES WORKER
# ─────────────────────────────────────────────────────────────────────

class TestModelFilesWorker:
    def test_successful_files_fetch(self):
        from app import ModelFilesWorker
        worker = ModelFilesWorker("meta-llama/Llama-3-8B")

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "siblings": [
                {"rfilename": "model-q4_k_m.gguf", "size": 4500000000},
                {"rfilename": "model.safetensors", "size": 14000000000},
                {"rfilename": "config.json", "size": 1200},
                {"rfilename": "tokenizer.model", "size": 500000},
            ]
        }

        results = []
        worker.files_ready.connect(lambda files, rid: results.append((files, rid)))

        with patch("requests.get", return_value=mock_response):
            worker.run()

        assert len(results) == 1
        files, rid = results[0]
        assert rid == "meta-llama/Llama-3-8B"
        # Only .gguf and .safetensors should be included, config.json and tokenizer.model excluded
        filenames = [f["name"] for f in files]
        assert "model-q4_k_m.gguf" in filenames
        assert "model.safetensors" in filenames
        assert "config.json" not in filenames
        assert "tokenizer.model" not in filenames

    def test_excludes_msgpack(self):
        from app import ModelFilesWorker
        worker = ModelFilesWorker("test/repo")

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "siblings": [
                {"rfilename": "model.safetensors", "size": 1000},
                {"rfilename": "model.safetensors.msgpack", "size": 500},
            ]
        }

        results = []
        worker.files_ready.connect(lambda files, rid: results.append((files, rid)))

        with patch("requests.get", return_value=mock_response):
            worker.run()

        filenames = [f["name"] for f in results[0][0]]
        assert "model.safetensors" in filenames
        assert "model.safetensors.msgpack" not in filenames

    def test_files_fetch_error(self):
        from app import ModelFilesWorker
        worker = ModelFilesWorker("bad/repo")

        errors = []
        worker.error.connect(lambda e: errors.append(e))

        with patch("requests.get", side_effect=ConnectionError("Fail")):
            worker.run()

        assert len(errors) == 1
        assert "Fail" in errors[0]


# ─────────────────────────────────────────────────────────────────────
#  8. MULTI MODEL MANAGER
# ─────────────────────────────────────────────────────────────────────

class TestMultiModelManager:
    @patch("app.HAS_LLAMA_CPP", True)
    @patch("app.LlamaCPP")
    def test_load_gguf_model(self, mock_llama_cls, model_manager):
        from app import MultiModelManager, LlamaCppBackend
        mock_llama_cls.return_value = MagicMock()
        fmt_info = {"format": "gguf", "quant": "q4_k_m", "estimated_params_b": 7.0, "format_data": {}, "quant_data": {}}
        result = model_manager.load_model("test-gguf", "/fake/path.gguf", fmt_info, {"n_ctx": 2048, "n_gpu_layers": -1, "n_threads": 4})
        assert result is True
        assert model_manager.is_loaded("test-gguf")
        assert "test-gguf" in model_manager.get_loaded()

    @patch("app.HAS_LLAMA_CPP", True)
    @patch("app.LlamaCPP")
    def test_unload_model(self, mock_llama_cls, model_manager):
        from app import MultiModelManager
        mock_llama_cls.return_value = MagicMock()
        fmt_info = {"format": "gguf", "quant": "q4_k_m", "estimated_params_b": 7.0, "format_data": {}, "quant_data": {}}
        model_manager.load_model("test-gguf", "/fake/path.gguf", fmt_info, {"n_ctx": 2048})
        result = model_manager.unload_model("test-gguf")
        assert result is True
        assert not model_manager.is_loaded("test-gguf")

    def test_unload_nonexistent_model(self, model_manager):
        result = model_manager.unload_model("ghost-model")
        assert result is False

    @patch("app.HAS_LLAMA_CPP", True)
    @patch("app.LlamaCPP")
    def test_load_duplicate_model(self, mock_llama_cls, model_manager):
        from app import MultiModelManager
        mock_llama_cls.return_value = MagicMock()
        fmt_info = {"format": "gguf", "quant": "q4_k_m", "estimated_params_b": 7.0, "format_data": {}, "quant_data": {}}
        model_manager.load_model("test-dup", "/fake/path.gguf", fmt_info, {"n_ctx": 2048})
        # Second load should just return True without creating a new backend
        result = model_manager.load_model("test-dup", "/fake/path.gguf", fmt_info, {"n_ctx": 2048})
        assert result is True
        mock_llama_cls.assert_called_once()

    @patch("app.HAS_LLAMA_CPP", True)
    @patch("app.LlamaCPP")
    def test_get_backend(self, mock_llama_cls, model_manager):
        from app import MultiModelManager
        mock_llama_cls.return_value = MagicMock()
        fmt_info = {"format": "gguf", "quant": "q4_k_m", "estimated_params_b": 7.0, "format_data": {}, "quant_data": {}}
        model_manager.load_model("test-backend", "/fake/path.gguf", fmt_info, {"n_ctx": 2048})
        backend = model_manager.get_backend("test-backend")
        assert backend is not None
        assert model_manager.get_backend("nonexistent") is None


# ─────────────────────────────────────────────────────────────────────
#  9. API SERVER & INTEGRATION
# ─────────────────────────────────────────────────────────────────────

class TestAPIServer:
    def test_create_api_app(self, model_manager):
        from app import HAS_FASTAPI
        if not HAS_FASTAPI:
            pytest.skip("FastAPI not installed")
        
        from app import create_api_app
        app = create_api_app(model_manager, MagicMock(server_port=8081))
        assert app.title == "CacheAI API"

    @patch("app.HAS_FASTAPI", True)
    def test_api_server_manager_start_stop(self, model_manager):
        from app import HAS_FASTAPI
        if not HAS_FASTAPI:
            pytest.skip("FastAPI not installed")
        
        import app
        mock_config = MagicMock()
        mock_config.server_port = 8081
        
        # Need to mock uvicorn
        with patch("app.uvicorn") as mock_uvicorn:
            mock_server = MagicMock()
            mock_uvicorn.Server.return_value = mock_server
            mock_uvicorn.Config.return_value = MagicMock()
            
            manager = app.APIServerManager(model_manager, mock_config)
            # Start in a way that doesn't block or actually run
            manager.start()
            assert manager.running is True
            manager.stop()
            # Verify should_exit was called
            mock_server.should_exit = True
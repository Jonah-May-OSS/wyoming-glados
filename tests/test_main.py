import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Path to project root: tests/.. = project root
ROOT = Path(__file__).resolve().parent.parent
MAIN_PATH = ROOT / "__main__.py"

assert MAIN_PATH.exists(), f"Cannot find __main__.py at: {MAIN_PATH}"

# Load module from file
spec = importlib.util.spec_from_file_location("glados_main", MAIN_PATH)
mainmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mainmod)


# ============================================================
# setup_logging tests
# ============================================================
def test_setup_logging_configures_handler_and_level(capsys):
    mainmod.setup_logging(debug=True, log_format="%(message)s")

    logger = logging.getLogger()

    # Debug mode sets DEBUG
    assert logger.level == logging.DEBUG

    # Handler should be StreamHandler using NanosecondFormatter
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0].formatter, mainmod.NanosecondFormatter)

    logger.debug("hello")

    captured = capsys.readouterr()
    assert "hello" in captured.out


# ============================================================
# run() should call asyncio.run(main)
# ============================================================
def test_run_calls_asyncio_run():
    with patch("asyncio.run") as run_mock:
        mainmod.run()
        run_mock.assert_called_once()


# ============================================================
# main() integration tests using full mocking
# ============================================================


@pytest.mark.asyncio
async def test_main_happy_path(monkeypatch):
    """Ensure main() hits the server.start() logic without errors."""
    # Fake arguments
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--uri=tcp://localhost:1234",
            "--models-dir=/tmp/models",
        ],
    )

    # -----------------------------------
    # Mock subprocess.run for model download
    # -----------------------------------
    mock_subproc = MagicMock()
    monkeypatch.setattr(mainmod.subprocess, "run", mock_subproc)

    # -----------------------------------
    # Mock TTSRunner
    # -----------------------------------
    mock_tts = MagicMock()
    monkeypatch.setattr(mainmod, "TTSRunner", MagicMock(return_value=mock_tts))

    # Mock rnn flatten_parameters call
    mock_tts.glados.rnn.flatten_parameters = MagicMock()

    # -----------------------------------
    # Mock NLTK lookup + download
    # -----------------------------------
    monkeypatch.setattr(mainmod.nltk_data, "find", MagicMock())
    monkeypatch.setattr(mainmod.nltk, "download", MagicMock())

    # -----------------------------------
    # Mock GladosProcessManager
    # -----------------------------------
    mock_proc_mgr = MagicMock()
    mock_proc_mgr.get_process = MagicMock(return_value=asyncio.sleep(0))
    monkeypatch.setattr(
        mainmod,
        "GladosProcessManager",
        MagicMock(return_value=mock_proc_mgr),
    )

    # -----------------------------------
    # Mock AsyncServer
    # -----------------------------------
    mock_server = MagicMock()
    mock_server.run = MagicMock(return_value=asyncio.sleep(0))
    monkeypatch.setattr(
        mainmod.AsyncServer, "from_uri", MagicMock(return_value=mock_server)
    )

    # -----------------------------------
    # Run main()
    # -----------------------------------
    await mainmod.main()

    # Ensure model download was triggered
    assert mock_subproc.called

    # Ensure server was started
    mock_server.run.assert_called_once()


@pytest.mark.asyncio
async def test_main_download_failure(monkeypatch, capsys):
    """Simulate a model download failure and ensure error is logged."""
    monkeypatch.setattr(sys, "argv", ["prog", "--models-dir=/tmp/models"])

    # subprocess.run → raise CalledProcessError
    def raise_err(*a, **k):
        raise mainmod.subprocess.CalledProcessError(1, "cmd")

    monkeypatch.setattr(mainmod.subprocess, "run", raise_err)

    # Mock the minimum objects needed for main() to proceed
    monkeypatch.setattr(mainmod, "TTSRunner", MagicMock())
    monkeypatch.setattr(mainmod.nltk_data, "find", MagicMock())
    monkeypatch.setattr(mainmod.nltk, "download", MagicMock())

    mock_proc_mgr = MagicMock()
    mock_proc_mgr.get_process = MagicMock(return_value=asyncio.sleep(0))
    monkeypatch.setattr(
        mainmod, "GladosProcessManager", MagicMock(return_value=mock_proc_mgr)
    )

    mock_server = MagicMock()
    mock_server.run = MagicMock(return_value=asyncio.sleep(0))
    monkeypatch.setattr(
        mainmod.AsyncServer, "from_uri", MagicMock(return_value=mock_server)
    )

    await mainmod.main()

    captured = capsys.readouterr()
    assert "Model download failed" in captured.out


@pytest.mark.asyncio
async def test_main_rnn_flatten_failure(monkeypatch):
    """flatten_parameters should be ignored if it throws."""
    monkeypatch.setattr(sys, "argv", ["prog"])

    # Mock subprocess
    monkeypatch.setattr(mainmod.subprocess, "run", MagicMock())

    # TTSRunner mock where flatten_parameters throws
    mock_tts = MagicMock()

    def raise_flatten():
        raise RuntimeError("fail")

    mock_tts.glados.rnn.flatten_parameters = raise_flatten
    monkeypatch.setattr(mainmod, "TTSRunner", MagicMock(return_value=mock_tts))

    # Other required mocks
    monkeypatch.setattr(mainmod.nltk_data, "find", MagicMock())
    monkeypatch.setattr(mainmod.nltk, "download", MagicMock())

    mock_proc_mgr = MagicMock()
    mock_proc_mgr.get_process = MagicMock(return_value=asyncio.sleep(0))
    monkeypatch.setattr(
        mainmod, "GladosProcessManager", MagicMock(return_value=mock_proc_mgr)
    )

    mock_server = MagicMock()
    mock_server.run = MagicMock(return_value=asyncio.sleep(0))
    monkeypatch.setattr(
        mainmod.AsyncServer, "from_uri", MagicMock(return_value=mock_server)
    )

    # Should not raise
    await mainmod.main()


@pytest.mark.asyncio
async def test_main_downloads_nltk_if_missing(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])

    # Mock subprocess
    monkeypatch.setattr(mainmod.subprocess, "run", MagicMock())

    # Mock TTS
    mock_tts = MagicMock()
    monkeypatch.setattr(mainmod, "TTSRunner", MagicMock(return_value=mock_tts))

    # Force NLTK lookup failure
    monkeypatch.setattr(mainmod.nltk_data, "find", MagicMock(side_effect=LookupError()))
    mock_download = MagicMock()
    monkeypatch.setattr(mainmod.nltk, "download", mock_download)

    # Process manager mocks
    mock_proc_mgr = MagicMock()
    mock_proc_mgr.get_process = MagicMock(return_value=asyncio.sleep(0))
    monkeypatch.setattr(
        mainmod, "GladosProcessManager", MagicMock(return_value=mock_proc_mgr)
    )

    # Server mock
    mock_server = MagicMock()
    mock_server.run = MagicMock(return_value=asyncio.sleep(0))
    monkeypatch.setattr(
        mainmod.AsyncServer, "from_uri", MagicMock(return_value=mock_server)
    )

    await mainmod.main()

    mock_download.assert_called_once()


@pytest.mark.asyncio
async def test_main_server_run_exception(monkeypatch, capsys):
    """Ensure exception inside server.run() triggers error log and sys.exit(1)."""

    # Fake CLI args
    monkeypatch.setattr(sys, "argv", ["prog"])

    # Mock subprocess.run (model downloader)
    monkeypatch.setattr(mainmod.subprocess, "run", MagicMock())

    # Mock TTSRunner
    mock_tts = MagicMock()
    monkeypatch.setattr(mainmod, "TTSRunner", MagicMock(return_value=mock_tts))

    # NLTK mocks
    monkeypatch.setattr(mainmod.nltk_data, "find", MagicMock())
    monkeypatch.setattr(mainmod.nltk, "download", MagicMock())

    # Mock process manager
    mock_proc_mgr = MagicMock()
    mock_proc_mgr.get_process = MagicMock(return_value=asyncio.sleep(0))
    monkeypatch.setattr(
        mainmod, "GladosProcessManager", MagicMock(return_value=mock_proc_mgr)
    )

    # Mock AsyncServer, but force .run() to throw
    mock_server = MagicMock()
    mock_server.run = MagicMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(
        mainmod.AsyncServer, "from_uri", MagicMock(return_value=mock_server)
    )

    # Execute and EXPECT SystemExit(1)
    with pytest.raises(SystemExit) as excinfo:
        await mainmod.main()

    # Ensure exit code = 1
    assert excinfo.value.code == 1

    # Verify the log message was printed (your logging writes to stdout)
    captured = capsys.readouterr()
    assert "An error occurred while running the server" in captured.out
    assert "boom" in captured.out

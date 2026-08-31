"""Unit tests for the top-level __main__ server entrypoint."""

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
assert spec is not None
assert spec.loader is not None
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
        # run() builds the coroutine eagerly and hands it to asyncio.run, which
        # is mocked here -- so nothing ever awaits it. Assert it is the right
        # one, then close it; otherwise it is collected at an arbitrary later
        # point and pytest reports "coroutine 'main' was never awaited" against
        # whichever test happens to be running then.
        (coro,) = run_mock.call_args.args
        assert coro.__name__ == "main"
        coro.close()


# ============================================================
# main() integration tests using full mocking
# ============================================================


@pytest.fixture(autouse=True)
def _no_real_download(monkeypatch):
    """Stop main() shelling out to download.py for real.

    Without this, every main() test spawned an actual
    `python download.py --model-dir ...` subprocess: real network access, real
    writes to whatever --models-dir the test passed, and a multi-second delay.
    It also made the download-failure branch untestable, since a genuinely
    failing subprocess was indistinguishable from the behaviour under test.
    """
    monkeypatch.setattr(
        mainmod.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0))
    )


def _stub_serving(monkeypatch):
    """Mock out everything main() needs after the download step."""
    monkeypatch.setattr(mainmod, "PiperTTSRunner", MagicMock(return_value=MagicMock()))
    proc_mgr = MagicMock()
    # side_effect, not return_value: return_value builds ONE coroutine and
    # hands the same object to every call, so a second call awaits an already
    # consumed coroutine and the first leaks a "never awaited" warning.
    proc_mgr.get_process = MagicMock(side_effect=lambda *a, **k: asyncio.sleep(0))
    monkeypatch.setattr(
        mainmod, "GladosProcessManager", MagicMock(return_value=proc_mgr)
    )
    server = MagicMock()
    server.run = MagicMock(side_effect=lambda *a, **k: asyncio.sleep(0))
    monkeypatch.setattr(mainmod.AsyncServer, "from_uri", MagicMock(return_value=server))
    return server


@pytest.mark.asyncio
async def test_exits_when_the_download_fails_and_no_voice_is_present(
    monkeypatch, tmp_path
):
    """No voice on disk and no download means there is nothing to serve.

    Continuing would raise FileNotFoundError out of PiperTTSRunner, reporting a
    missing file rather than the download failure that caused it.
    """
    monkeypatch.setattr(
        sys, "argv", ["prog", f"--models-dir={tmp_path}", "--voice-name=glados"]
    )
    monkeypatch.setattr(
        mainmod.subprocess,
        "run",
        MagicMock(side_effect=mainmod.subprocess.CalledProcessError(1, "download.py")),
    )
    _stub_serving(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        await mainmod.main()
    assert excinfo.value.code == 1


@pytest.mark.asyncio
async def test_exits_when_only_half_the_voice_is_present(monkeypatch, tmp_path):
    """A graph with no config is not a usable voice.

    _load reads the .json unguarded, so proceeding here dies with a bare
    traceback about a missing file - the failure this branch prevents.
    """
    (tmp_path / "glados.onnx").write_bytes(b"0" * 2048)  # config never arrived
    monkeypatch.setattr(
        sys, "argv", ["prog", f"--models-dir={tmp_path}", "--voice-name=glados"]
    )
    monkeypatch.setattr(
        mainmod.subprocess,
        "run",
        MagicMock(side_effect=mainmod.subprocess.CalledProcessError(1, "download.py")),
    )
    _stub_serving(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        await mainmod.main()
    assert excinfo.value.code == 1


@pytest.mark.asyncio
async def test_serves_the_existing_voice_when_the_download_fails(monkeypatch, tmp_path):
    """A failed refresh must not take a working offline deployment down."""
    # Both files: the runner needs the config for the phoneme_id_map, so a
    # voice is only usable when the graph AND the .json are present.
    (tmp_path / "glados.onnx").write_bytes(b"0" * 2048)
    (tmp_path / "glados.onnx.json").write_text("{}")
    monkeypatch.setattr(
        sys, "argv", ["prog", f"--models-dir={tmp_path}", "--voice-name=glados"]
    )
    monkeypatch.setattr(
        mainmod.subprocess,
        "run",
        MagicMock(side_effect=mainmod.subprocess.CalledProcessError(1, "download.py")),
    )
    server = _stub_serving(monkeypatch)

    # No SystemExit, and the server still comes up. Asserted on behaviour
    # rather than log text: main() installs its own logging handler, so the
    # warning never reaches caplog.
    await mainmod.main()

    server.run.assert_called_once()


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

    mock_tts = MagicMock()
    monkeypatch.setattr(mainmod, "PiperTTSRunner", MagicMock(return_value=mock_tts))

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

    # Ensure server was started
    mock_server.run.assert_called_once()


@pytest.mark.asyncio
async def test_main_server_run_exception(monkeypatch, capsys):
    """Ensure exception inside server.run() triggers error log and sys.exit(1)."""

    # Fake CLI args
    monkeypatch.setattr(sys, "argv", ["prog"])

    mock_tts = MagicMock()
    monkeypatch.setattr(mainmod, "PiperTTSRunner", MagicMock(return_value=mock_tts))

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


@pytest.mark.asyncio
async def test_warmup_completes_before_the_server_binds(monkeypatch):
    """Warmup is synchronous, and the ordering it implies is the point.

    This test used to assert the opposite - that warmup was handed to a daemon
    thread so the server could bind first. That never bought anything: the
    TensorRT engines are built inside PiperTTSRunner(), which has already
    returned by the time warmup runs, so the thread deferred a 0.03s smoke test
    while the ~60s cold build stayed on the startup path regardless.

    What matters is that warmup has run before the server accepts a connection,
    so the first request never races an unproven session.
    """
    monkeypatch.setattr(sys, "argv", ["prog"])
    mock_tts = MagicMock()
    monkeypatch.setattr(mainmod, "PiperTTSRunner", MagicMock(return_value=mock_tts))

    order = []
    mock_tts.warmup.side_effect = lambda: order.append("warmup")

    mock_proc_mgr = MagicMock()
    mock_proc_mgr.get_process = MagicMock(return_value=asyncio.sleep(0))
    monkeypatch.setattr(
        mainmod, "GladosProcessManager", MagicMock(return_value=mock_proc_mgr)
    )
    mock_server = MagicMock()
    mock_server.run = MagicMock(return_value=asyncio.sleep(0))

    def from_uri(*_args, **_kwargs):
        order.append("bind")
        return mock_server

    monkeypatch.setattr(mainmod.AsyncServer, "from_uri", from_uri)

    await mainmod.main()

    mock_tts.warmup.assert_called_once()
    assert order == ["warmup", "bind"], (
        "warmup must finish before the server binds, so no request can arrive "
        "against an unproven session"
    )


def test_warmup_failure_does_not_kill_the_server(caplog):
    """A failed smoke test is logged, not fatal: other providers still work."""
    runner = MagicMock()
    runner.warmup.side_effect = RuntimeError("no engine")
    with caplog.at_level(logging.ERROR):
        mainmod._warmup(runner)
    assert "warmup failed" in caplog.text.lower()


@pytest.mark.asyncio
async def test_advertised_voice_name_follows_the_loaded_voice(monkeypatch):
    """A hardcoded name would drift from --voice-name and confuse clients."""
    monkeypatch.setattr(sys, "argv", ["prog", "--voice-name=testvoice"])
    monkeypatch.setattr(mainmod, "PiperTTSRunner", MagicMock())

    captured = {}
    real_info = mainmod.Info

    def capture(**kwargs):
        captured.update(kwargs)
        return real_info(**kwargs)

    monkeypatch.setattr(mainmod, "Info", capture)

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

    voices = captured["tts"][0].voices
    assert [v.name for v in voices] == ["testvoice"]


@pytest.mark.asyncio
async def test_attribution_credits_this_project(monkeypatch):
    """The VITS voice is trained here, not by the ForwardTacotron project."""
    monkeypatch.setattr(sys, "argv", ["prog"])
    monkeypatch.setattr(mainmod, "PiperTTSRunner", MagicMock())

    captured = {}
    real_info = mainmod.Info

    def capture(**kwargs):
        captured.update(kwargs)
        return real_info(**kwargs)

    monkeypatch.setattr(mainmod, "Info", capture)

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

    attribution = captured["tts"][0].voices[0].attribution
    assert "R2D2FISH" not in attribution.name
    assert "wyoming-glados" in attribution.url

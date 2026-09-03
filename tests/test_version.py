"""The release workflow owns the version, so main does not carry a number.

No GPU and no model: this is a file and a regex. Before this, the version
lived only in ``pyproject.toml``, where it read ``0.1.0`` through every
release from 1.0.2 to 2.0.0 because nobody had a reason to touch it -- and
nothing read it, so nothing noticed.

The number now reaches Home Assistant, via the ``version`` field of the
Wyoming ``TtsProgram`` info message, which is what makes a stale one worth a
test.
"""

import importlib.util
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOW = _ROOT / ".github" / "workflows" / "release.yaml"
_VERSION_FILE = _ROOT / "VERSION"

_PLACEHOLDER = "0.0.0.dev0"

# Every tag the repository has actually released. The pattern in the workflow
# rejects the branch outright, so validating it against invented examples only
# would prove it accepts what I imagined rather than what we ship.
_RELEASED = [
    "1.0.2",
    "1.0.3",
    "1.0.4",
    "1.0.5",
    "1.0.6",
    "1.0.7",
    "1.0.8",
    "1.0.9",
    "1.0.10",
    "1.1.0",
    "2.0.0",
]

# This repository tags things that are not releases of it. None of these may
# be accepted as a release branch name: pushing release/voice-2026.09.01 must
# fail in the workflow rather than cut a release named after a voice.
_NOT_RELEASES = [
    "voice-2026.09.01",
    "ort-aarch64-v1.29.0",
    "backup/pre-rebranch-20260831",
]


def _release_version_pattern() -> str:
    """Return the pattern the workflow validates a release branch name with.

    Read out of the workflow rather than copied, so the two cannot drift apart
    silently. It is an ERE fed to ``grep -E``; the constructs used here mean
    the same thing to Python's ``re``.
    """
    text = _WORKFLOW.read_text(encoding="utf-8")
    match = re.search(r"VERSION_PATTERN='([^']+)'", text)
    assert match, f"no VERSION_PATTERN assignment in {_WORKFLOW.name}"
    return match.group(1)


def test_main_carries_a_placeholder_rather_than_a_release_number() -> None:
    """A real number here is a hand-edit the release branch will overwrite."""
    assert _VERSION_FILE.read_text(encoding="utf-8").strip() == _PLACEHOLDER


def test_the_placeholder_is_not_mistaken_for_a_release_version() -> None:
    assert not re.match(_release_version_pattern(), _PLACEHOLDER)


def test_the_server_advertises_what_the_version_file_says() -> None:
    """The wiring, not the file.

    Writing VERSION is only useful if the running server reports it, so this
    loads __main__ the way tests/test_main.py does and checks the value it
    exposes comes from the file the workflow writes.
    """
    spec = importlib.util.spec_from_file_location(
        "glados_main_version", _ROOT / "__main__.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__version__ == _VERSION_FILE.read_text(encoding="utf-8").strip()


@pytest.mark.parametrize("version", _RELEASED)
def test_every_version_released_so_far_is_accepted(version: str) -> None:
    assert re.match(_release_version_pattern(), version)


@pytest.mark.parametrize("version", ["rc1", "1.2", "v1.2.3", "1.2.3.4", ""])
def test_a_branch_that_does_not_name_a_version_is_rejected(version: str) -> None:
    """release/<anything> is pushable; the tag and the package name come from it."""
    assert not re.match(_release_version_pattern(), version)


@pytest.mark.parametrize("tag", _NOT_RELEASES)
def test_tags_that_are_not_releases_of_this_package_are_rejected(tag: str) -> None:
    assert not re.match(_release_version_pattern(), tag)

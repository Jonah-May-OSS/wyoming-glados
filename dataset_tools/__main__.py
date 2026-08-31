"""CLI for building the GLaDOS training corpus.

python -m dataset_tools fetch --out data
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from .build import MAX_SECONDS, MIN_LOW_MID_DB, MIN_SECONDS, build_dataset
from .fetch import (
    DEFAULT_DELAY,
    fetch_audio,
    fetch_pages,
    find_filename_collisions,
    http_opener,
)


def _fetch(args: argparse.Namespace) -> int:
    out = Path(args.out)
    opener = http_opener()

    print("Fetching wiki pages...")
    lines = fetch_pages(out / "pages", opener=opener, delay=args.delay)
    by_speaker = collections.Counter(line.speaker for line in lines)
    print(f"  {len(lines)} lines: {dict(by_speaker)}")

    collisions = find_filename_collisions(lines)
    if collisions:
        print(f"  WARNING: {len(collisions)} filename collisions", file=sys.stderr)
        for name, urls in list(collisions.items())[:5]:
            print(f"    {name}: {urls}", file=sys.stderr)

    if args.limit:
        lines = lines[: args.limit]
        print(f"  limited to {len(lines)} lines")

    print(f"Fetching audio into {out / 'audio'} ...")

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    report = fetch_audio(
        lines,
        out / "audio",
        opener=opener,
        delay=args.delay,
        on_progress=progress,
    )
    print(
        f"Done: {report.downloaded} downloaded, {report.skipped} cached, "
        f"{len(report.failures)} failed"
    )
    for url, error in report.failures[:20]:
        print(f"  FAILED {url}: {error}", file=sys.stderr)
    return 0 if report.ok else 1


def _build(args: argparse.Namespace) -> int:
    out = Path(args.out)
    lines = fetch_pages(out / "pages", opener=http_opener(), delay=args.delay)
    print(f"Building dataset from {len(lines)} lines...")
    report = build_dataset(
        lines,
        out / "audio",
        Path(args.dataset),
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        trim=not args.no_trim,
        normalize=not args.no_normalize,
        multi_speaker=args.multi_speaker,
        potato_speaker=args.potato_speaker,
        min_low_mid_db=args.min_low_mid_db,
    )
    print(report.summary())
    print()
    print(f"Wrote {Path(args.dataset) / 'metadata.csv'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the dataset CLI."""
    parser = argparse.ArgumentParser(prog="dataset_tools")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download wiki pages and audio")
    fetch.add_argument("--out", default="data", help="output directory")
    fetch.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="seconds between requests",
    )
    fetch.add_argument(
        "--limit", type=int, default=0, help="only fetch the first N lines"
    )
    fetch.set_defaults(func=_fetch)

    build = sub.add_parser("build", help="normalize audio and write metadata.csv")
    build.add_argument("--out", default="data", help="fetch output directory")
    build.add_argument(
        "--dataset", default="data/dataset", help="dataset output directory"
    )
    build.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    build.add_argument("--min-seconds", type=float, default=MIN_SECONDS)
    build.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    build.add_argument("--no-trim", action="store_true", help="keep silence")
    build.add_argument(
        "--no-normalize", action="store_true", help="keep original levels"
    )
    build.add_argument(
        "--multi-speaker",
        action="store_true",
        help="emit one speaker ID per source instead of pooling into one voice",
    )
    build.add_argument(
        "--potato-speaker",
        action="store_true",
        help=(
            "keep the band-pass filtered potato-battery clips as their own "
            "speaker instead of dropping them (requires --multi-speaker)"
        ),
    )
    build.add_argument(
        "--min-low-mid-db",
        type=float,
        default=MIN_LOW_MID_DB,
        help=(
            "drop clips whose low/mid band energy ratio is below this; "
            "identifies band-pass-filtered audio such as the potato-battery "
            "and PA-speaker scenes"
        ),
    )
    build.set_defaults(func=_build)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create a lightweight OpenPI inference bundle around a downloaded params-only checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-dir", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset-id", default="so101_grab_blue_pen_60")
    return parser.parse_args()


def relative_symlink(source: Path, destination: Path) -> None:
    destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=source.is_dir())


def main() -> None:
    args = parse_args()
    params_dir = args.params_dir.expanduser().resolve()
    norm_stats = args.norm_stats.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not (params_dir / "_METADATA").is_file():
        raise FileNotFoundError(f"Not an Orbax params directory: {params_dir}")
    if not norm_stats.is_file():
        raise FileNotFoundError(norm_stats)
    payload = json.loads(norm_stats.read_text())
    stats = payload.get("norm_stats", payload)
    for key in ("state", "actions"):
        if key not in stats:
            raise ValueError(f"Missing {key!r} normalization stats in {norm_stats}")
        for field in ("mean", "std", "q01", "q99"):
            if len(stats[key][field]) != 6:
                raise ValueError(f"Expected six {key}.{field} values, got {len(stats[key][field])}")

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {output_dir}")
    asset_dir = output_dir / "assets" / args.asset_id
    asset_dir.mkdir(parents=True)
    relative_symlink(params_dir, output_dir / "params")
    relative_symlink(norm_stats, asset_dir / "norm_stats.json")
    manifest = {
        "config": "pi05_so101_60",
        "params": str(params_dir),
        "norm_stats": str(norm_stats),
        "asset_id": args.asset_id,
        "note": "Symlink-only local inference bundle; no model or dataset data was copied.",
    }
    (output_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created OpenPI inference bundle: {output_dir}")
    print(f"Serve with: uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_so101_60 --policy.dir={output_dir}")


if __name__ == "__main__":
    main()

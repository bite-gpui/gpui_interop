#!/usr/bin/env python3
"""Stage this workspace as a publishable tree.

During development the core comes from the sibling `bite-gpui` checkout as path
dependencies. The registry knows the same crates under the distribution's names
(`bite-gpui`, `bite-gp-engine`, `bite-gp-wgpu`, `bite-gp-windows`), and Cargo
cannot rename a *path* package: the package at the path is named `gpui`, so a
`package = "bite-gpui"` entry would look for a package that is not there.

This copies the workspace to a destination, rewrites those four dependency lines
to the published packages, and drops the development-only `[patch.crates-io]`
table. The crate itself is copied unchanged. Run `cargo publish` in the staged
tree; CI uses the same script for its package check.

    python3 scripts/prepare-publish.py /path/to/staged-tree
    (cd /path/to/staged-tree && cargo publish --dry-run -p gpui_interop)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REPLACEMENTS = [
    (
        'gpui = { path = "../bite-gpui/crates/gpui", default-features = false }',
        'gpui = { package = "bite-gpui", version = "1.24", default-features = false }',
    ),
    (
        'gpui_engine = { path = "../bite-gpui/crates/gpui_engine" }',
        'gpui_engine = { package = "bite-gp-engine", version = "1.24" }',
    ),
    (
        'gpui_wgpu = { path = "../bite-gpui/crates/gpui_wgpu" }',
        'gpui_wgpu = { package = "bite-gp-wgpu", version = "1.24" }',
    ),
    (
        'gpui_windows = { path = "../bite-gpui/crates/gpui_windows" }',
        'gpui_windows = { package = "bite-gp-windows", version = "1.24" }',
    ),
]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} DESTINATION")

    dest = Path(sys.argv[1]).resolve()
    if dest == ROOT or ROOT in dest.parents:
        raise SystemExit(f"refusing to stage inside the source tree: {dest}")
    if dest.exists():
        shutil.rmtree(dest)

    shutil.copytree(
        ROOT,
        dest,
        ignore=shutil.ignore_patterns(".git", "target", "Cargo.lock"),
    )

    manifest = dest / "Cargo.toml"
    text = manifest.read_text()
    for old, new in REPLACEMENTS:
        if old not in text:
            raise SystemExit(f"expected dependency line not found: {old}")
        text = text.replace(old, new)

    # The patch table exists for the development core's forks. The published core
    # resolves its own dependencies from crates.io; leaving the patch in would hide
    # whether the package really builds without it.
    marker = "\n[patch.crates-io]"
    if marker in text:
        text = text[: text.index(marker)] + "\n"

    manifest.write_text(text)
    print(dest)


if __name__ == "__main__":
    main()

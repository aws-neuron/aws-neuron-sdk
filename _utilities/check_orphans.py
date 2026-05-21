#!/usr/bin/env python3
"""Precise inbound-reference inventory for a list of orphaned Sphinx docs.

For each orphan, we look for:
  1. `:doc:` directives referencing the orphan path:
        :doc:`label <path>`            (any path form)
        :doc:`path`
     We accept both leading-slash and bare paths, and also bare basenames
     when they appear inside a `:doc:` role.
  2. `:ref:` directives referencing any `.. _label:` declared in the orphan.
  3. Toctree entries (a line inside a `.. toctree::` block whose value
     matches the orphan path or basename).
  4. `:download:` directives referencing the path (for completeness).

Substring matching of bare paths is intentionally avoided because basenames
like `index`, `features`, or `nrt` produce massive false-positive sets.

Output: a markdown report grouped by status (live / orphan-only / none).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_DIRS = {"_build", "_backup-rn", "_backup-setup", ".venv", ".git",
                "node_modules"}
INCLUDE_EXTS = {".rst", ".ipynb", ".py", ".txt", ".md"}

ANCHOR_RE = re.compile(r"^\.\. _([A-Za-z0-9_\-\.]+):\s*$", re.MULTILINE)

ORPHANS = [
    "about-neuron/calculator/neuron-calculator.rst",
    "about-neuron/monitoring-tools.rst",
    "about-neuron/profiling-tools.rst",
    "about-neuron/quick-start/tab-inference-tensorflow-neuron.rst",
    "about-neuron/quick-start/torch-neuron-tab-training.rst",
    "about-neuron/quick-start/user-guide-quickstart.rst",
    "archive/profiler/index.rst",
    "archive/transformers-neuronx/transformers-neuronx-api-reference.rst",
    "archive/tutorials/gpt3_neuronx_nemo_megatron_pretraining.rst",
    "frameworks/torch/torch-neuronx/api-reference-guide/inference/api-torch-neuronx-async-lazy-load.rst",
    "frameworks/torch/torch-neuronx/programming-guide/inference/autobucketing-dev-guide.rst",
    "frameworks/torch/torch-neuronx/setup/prev-releases/neuronx-2.7.0-pytorch-install.rst",
    "frameworks/torch/torch-neuronx/setup/prev-releases/neuronx-2.8.0-pytorch-install.rst",
    "frameworks/torch/torch-neuronx/setup/prev-releases/neuronx-2.9.0-pytorch-install.rst",
    "frameworks/torch/torch-neuronx/setup/pytorch-install.rst",
    "frameworks/torch/torch-neuronx/setup/pytorch-install-prev-al2.rst",
    "frameworks/torch/torch-neuronx/setup/pytorch-install-prev-u20.rst",
    "frameworks/torch/torch-neuronx/setup/pytorch-neuronx-install-cxx11.rst",
    "frameworks/torch/torch-neuronx/setup/pytorch-update-al2.rst",
    "frameworks/torch/torch-neuronx/setup/pytorch-update-al2-dlami.rst",
    "frameworks/torch/torch-neuronx/setup/pytorch-update-u20.rst",
    "frameworks/torch/torch-neuronx/setup/pytorch-update-u20-dlami.rst",
    "libraries/index.rst",
    "libraries/neuronx-distributed/developer-guide.rst",
    "libraries/neuronx-distributed/standard_mixed_precision.rst",
    "libraries/neuronx-distributed/tutorials/inference.rst",
    "libraries/nxd-training/app_notes/nxd-training-cp-appnote.rst",
    "libraries/nxd-training/developer-guide.rst",
    "libraries/nxd-training/general/features.rst",
    "neuron-runtime/api/nrt.rst",
    "neuron-runtime/api/nrt-async-api-best-practices.rst",
    "neuron-runtime/api/nrt-async-api-examples.rst",
    "neuron-runtime/api/nrt-async-api-overview.rst",
    "neuron-runtime/api/nrt_status.rst",
    "neuron-runtime/api/nrt_version.rst",
    "neuron-runtime/rn.rst",
    "nki/api/generated/nki.language.shared_constant.rst",
    "nki/api/generated/nki.language.tile_size.rst",
    "nki/api/nki.simulate.rst",
    "nki/deep-dives/index.rst",
    "nki/get-started/index.rst",
    "nki/guides/index.rst",
    "nki/migration/index.rst",
    "release-notes/archive/neuron-cc/neuron-cc-ops/neuron-cc-ops-xla.rst",
    "release-notes/archive/neuron1/neuronrelease/previous-content.rst",
    "release-notes/archive/neuron1/prev/content.rst",
    "setup/install-templates/inf1/launch-inf1-dlami-aws-cli.rst",
    "setup/install-templates/inf1/neuron-pip-install.rst",
    "setup/jax-neuronx.rst",
    "setup/mxnet-neuron.rst",
    "setup/notebook/running-jupyter-notebook-as-script.rst",
    "setup/notebook/setup-jupyter-notebook-steps-troubleshooting.rst",
    "setup/setup-troubleshooting.rst",
    "setup/torch-neuron.rst",
    "setup/torch-neuron-ubuntu20.rst",
    "setup/torch-neuronx.rst",
    "src/examples/mxnet/data_parallel/data_parallel_tutorial.ipynb",
    "src/examples/mxnet/mxnet-gluon-tutorial.ipynb",
    "src/examples/mxnet/resnet50/resnet50.ipynb",
    "src/examples/mxnet/resnet50_neuroncore_groups.ipynb",
    "src/examples/pytorch/resnet50.ipynb",
    "src/examples/pytorch/resnet50_partition.ipynb",
    "src/examples/pytorch/yolo_v4.ipynb",
    "src/examples/tensorflow/keras_resnet50/keras_resnet50.ipynb",
    "src/examples/tensorflow/openpose_demo/openpose.ipynb",
    "src/examples/tensorflow/tensorflow_resnet50/resnet50.ipynb",
    "src/examples/tensorflow/yolo_v3_demo/yolo_v3.ipynb",
    "src/examples/tensorflow/yolo_v4_demo/evaluate.ipynb",
]


def iter_search_files():
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext not in INCLUDE_EXTS:
                continue
            yield Path(dirpath) / fn


def extract_anchors(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    return ANCHOR_RE.findall(text)


def build_patterns(rel: str, anchors: list[str]):
    """Construct precise regex patterns for inbound references to this orphan.

    Returns dict of category -> compiled pattern.
    """
    stem = rel.rsplit(".", 1)[0]                  # drop suffix
    bare = "/" + stem                              # leading-slash form
    nolead = stem
    basename = Path(stem).name

    # Path candidates we'll match inside :doc: / :download: / toctree.
    # Skip the bare basename when it's a generic name like "index" — too many
    # false positives (every dir has its own index).
    GENERIC_BASENAMES = {"index", "rn", "overview", "tutorials"}
    paths = [bare, nolead]
    if basename not in GENERIC_BASENAMES:
        paths.append(basename)
    paths_for_role = [re.escape(p) for p in paths]
    paths_alt = "(?:" + "|".join(paths_for_role) + ")"

    # :doc:`...` may have either bare path or "title <path>" form.
    # We allow either backtick variant by matching ``:doc:`...``` with
    # `paths_alt` either directly or inside angle brackets.
    doc_role = re.compile(
        r":doc:`[^`]*?(?:<\s*" + paths_alt + r"\s*>|\b" + paths_alt + r"\b)[^`]*?`"
    )
    download_role = re.compile(
        r":download:`[^`]*?(?:<\s*" + paths_alt + r"\s*>|\b" + paths_alt + r"\b)[^`]*?`"
    )
    # Toctree entry: line whose trimmed content equals one of the paths
    # (with optional title prefix using `<path>` syntax, which is more typical
    # of grid cards but also valid in toctree). We'll search line-by-line.
    toctree_line = re.compile(
        r"^\s+(?:[^<\n]*<\s*" + paths_alt + r"\s*>\s*|\s*" + paths_alt + r"\s*)$",
        re.MULTILINE,
    )

    ref_role = None
    if anchors:
        anchors_alt = "(?:" + "|".join(re.escape(a) for a in anchors) + ")"
        # :ref:`label` or :ref:`text <label>`
        ref_role = re.compile(
            r":ref:`[^`]*?(?:<\s*" + anchors_alt + r"\s*>|\b" + anchors_alt + r"\b)[^`]*?`"
        )

    return {
        "doc": doc_role,
        "download": download_role,
        "toctree": toctree_line,
        "ref": ref_role,
    }


def in_toctree(text: str, line_pattern: re.Pattern) -> list[str]:
    """Return matching toctree-style lines, but only when they fall inside
    a `.. toctree::` block. Returns the matched line text."""
    hits = []
    in_block = False
    block_indent = 0
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not in_block:
            if stripped.startswith(".. toctree::"):
                in_block = True
                block_indent = len(raw_line) - len(raw_line.lstrip())
            continue
        # Still inside toctree until we hit a blank line followed by a
        # less-indented non-blank line, or another directive at base indent.
        if not stripped:
            continue
        cur_indent = len(raw_line) - len(raw_line.lstrip())
        if cur_indent <= block_indent and not stripped.startswith(":"):
            in_block = False
            continue
        # Match within the block
        if line_pattern.search(raw_line + "\n"):
            hits.append(raw_line.strip())
    return hits


def main() -> None:
    abs_orphans = {REPO_ROOT / o for o in ORPHANS}

    # Build patterns
    per_orphan = {}
    for rel in ORPHANS:
        anchors = extract_anchors(REPO_ROOT / rel)
        per_orphan[rel] = {
            "anchors": anchors,
            "patterns": build_patterns(rel, anchors),
            "hits": {"doc": {}, "ref": {}, "toctree": {}, "download": {}},
        }

    # Single walk over the corpus
    for fp in iter_search_files():
        if fp in abs_orphans:
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for rel, data in per_orphan.items():
            pats = data["patterns"]
            d_hits = pats["doc"].findall(text)
            if d_hits:
                data["hits"]["doc"][fp] = d_hits
            dl_hits = pats["download"].findall(text)
            if dl_hits:
                data["hits"]["download"][fp] = dl_hits
            if pats["ref"] is not None:
                r_hits = pats["ref"].findall(text)
                if r_hits:
                    data["hits"]["ref"][fp] = r_hits
            t_hits = in_toctree(text, pats["toctree"])
            if t_hits:
                data["hits"]["toctree"][fp] = t_hits

    # Classify
    for rel, data in per_orphan.items():
        all_files = (
            set(data["hits"]["doc"]) | set(data["hits"]["ref"]) |
            set(data["hits"]["toctree"]) | set(data["hits"]["download"])
        )
        if not all_files:
            data["status"] = "none"
            continue
        live_files = [f for f in all_files if f not in abs_orphans]
        data["status"] = "live" if live_files else "orphan-only"

    print("# Orphan inbound-reference inventory\n")
    print(f"Repo root: `{REPO_ROOT}`\n")
    print(f"Total orphans inspected: {len(per_orphan)}\n")
    print("Search categories: `:doc:`, `:ref:`, `toctree` entries, "
          "`:download:`. Substring path matching is intentionally not used.\n")

    by_status = {"live": [], "orphan-only": [], "none": []}
    for rel, data in per_orphan.items():
        by_status[data["status"]].append((rel, data))

    for status, label in [
        ("live", "Referenced from live (non-orphan) content"),
        ("orphan-only", "Only referenced by other orphans"),
        ("none", "No precise inbound references found"),
    ]:
        bucket = by_status[status]
        print(f"## {label} ({len(bucket)})\n")
        if not bucket:
            print("_None._\n")
            continue
        for rel, data in bucket:
            print(f"### `{rel}`")
            if data["anchors"]:
                print(f"- Anchors: {', '.join('`%s`' % a for a in data['anchors'])}")
            else:
                print("- Anchors: _none_")
            for cat in ("doc", "ref", "toctree", "download"):
                hits = data["hits"][cat]
                if not hits:
                    continue
                cat_label = {
                    "doc": ":doc: hits",
                    "ref": ":ref: hits",
                    "toctree": "toctree entries",
                    "download": ":download: hits",
                }[cat]
                print(f"- {cat_label}:")
                for fp, samples in hits.items():
                    rel_fp = Path(fp).relative_to(REPO_ROOT)
                    sample = samples[0] if samples else ""
                    if isinstance(sample, tuple):
                        sample = " | ".join(s for s in sample if s)
                    sample = sample.strip()
                    if len(sample) > 120:
                        sample = sample[:117] + "..."
                    print(f"  - `{rel_fp}` — `{sample}`")
            print()


if __name__ == "__main__":
    main()

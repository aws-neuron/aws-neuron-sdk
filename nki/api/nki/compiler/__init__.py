# AUTO GENERATED FILE - DO NOT MANUALLY MODIFY.

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import *

class CompileConfig():
    r"""Per-kernel compile configuration passed to ``@nki.jit(compile_config=...)``.

    Options that impact downstream compiler behavior."""

    address_rotation: AddressRotation = ...
    r"""Backend address-rotation policy (an :class:`AddressRotation`)."""

    non_ssa_legalization: bool = ...
    r"""Enable the backend's non-SSA legalization pass on this kernel's memlocs."""

    dead_code_elim: bool = ...
    r"""Leave this kernel's memlocs eligible for backend dead-code elimination."""

    barrier_check: bool = ...
    r"""Compile with the LNC core barrier checker enabled."""

    prefetch_auto: bool = ...
    r"""Enable AUTO DMA-prefetch allocation: the compiler classifies load
    destinations automatically, reserves a fixed prefetch ring + slack band in
    SBUF, and lets compute/prefetch spill into the shared slack. Mutually
    exclusive with a manual ``NKI_PREFETCH_RING_SIZE``. Off by default."""

    nki_alloc_fallback: bool = ...
    r"""Fall back to the default neuronx-cc allocation flow when NKI's own
    LinearScanAllocation fails during compilation. On by default; set False to
    surface the failure (e.g. an SBUF/PSUM out-of-memory) instead of
    retrying."""

    ...

class SpecializationConfig():
    r"""Per-kernel specialization-time config passed to ``@nki.jit(specialization_config=...)``.

    Options that tune the specialization phase (turning the kernel into IR)."""

    allow_nested_config_override: bool = ...
    r"""When this kernel is the top-level compiled kernel, warn (instead of
    raising) if a nested @nki.jit sub-kernel's config disagrees, and let it inline
    under this kernel's config."""

    ...

class AddressRotation(Enum):
    r"""Whether the backend may rotate a memloc's address for better layout.

    - ``AUTO`` (default): decide per memloc by allocation origin — a memloc with
      a fixed address (``bind_memloc``, whether user-supplied or chosen by NKI's
      LinearScanAllocation) is opted OUT of rotation so the backend does not
      reshuffle an address the kernel already fixed; a memloc the backend
      allocates keeps rotation ON so it can still pick a good layout.
    - ``ENABLE``: keep rotation on for every memloc.
    - ``DISABLE``: opt every memloc out of rotation."""

    AUTO = 'auto'

    ENABLE = 'enable'

    DISABLE = 'disable'

    ...

"""Compatibility shim re-exporting ``sc_save.workflows.perturbation``."""

from __future__ import annotations

import sys

import sc_save.workflows.perturbation as _impl

sys.modules[__name__] = _impl

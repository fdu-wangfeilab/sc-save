"""Compatibility shim re-exporting ``sc_save.workflows.bec``."""

from __future__ import annotations

import sys

import sc_save.workflows.bec as _impl

sys.modules[__name__] = _impl

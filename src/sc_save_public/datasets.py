"""Backward-compatibility shim aliasing :mod:`sc_save.data.datasets`."""

from __future__ import annotations

import sys

import sc_save.data.datasets as _datasets_impl

sys.modules[__name__] = _datasets_impl

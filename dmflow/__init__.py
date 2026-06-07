"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

try:
    __version__ = package_version("dmflow")
except PackageNotFoundError:
    __version__ = "0.1.0"

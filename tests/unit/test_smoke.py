"""Smoke tests: verify the package is importable and CLI entry point exists."""

import importlib


def test_hyporeddit_package_importable() -> None:
    mod = importlib.import_module("hyporeddit")
    assert mod is not None


def test_hyporeddit_cli_importable() -> None:
    mod = importlib.import_module("hyporeddit.cli")
    assert hasattr(mod, "app")


def test_hyporeddit_config_importable() -> None:
    mod = importlib.import_module("hyporeddit.config")
    assert hasattr(mod, "settings")

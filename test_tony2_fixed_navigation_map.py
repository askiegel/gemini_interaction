#!/usr/bin/env python3

"""Integrity checks for Tony2's validated fixed Nav2 map."""

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent

ASSET_DIR = (
    ROOT
    / "voice_relay"
    / "tony2_navigation_assets"
)

MAP_YAML = (
    ASSET_DIR
    / "mayday_supervised_route_03.yaml"
)

MAP_PGM = (
    ASSET_DIR
    / "mayday_supervised_route_03.pgm"
)

EXPECTED_YAML_SHA256 = (
    "8b9d9b9aae7875a30b0611b0d88496c"
    "caafbf54c75be1d808eea61f90aeedf1f"
)

EXPECTED_PGM_SHA256 = (
    "e2cf598840f8fe65475205ba08d99106"
    "d4c48bcc91deff3e2df15ca138a84f51"
)


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_validated_fixed_map_assets_exist():
    assert MAP_YAML.is_file()
    assert MAP_PGM.is_file()

    assert MAP_YAML.stat().st_size > 0
    assert MAP_PGM.stat().st_size > 0


def test_validated_fixed_map_assets_are_exact():
    assert sha256(MAP_YAML) == EXPECTED_YAML_SHA256
    assert sha256(MAP_PGM) == EXPECTED_PGM_SHA256


def test_fixed_map_yaml_uses_local_pgm():
    text = MAP_YAML.read_text(
        encoding="utf-8"
    )

    required = (
        "image: mayday_supervised_route_03.pgm",
        "resolution: 0.050000",
        "origin: [-2.600000, -2.550000, 0.0]",
        "negate: 0",
        "occupied_thresh: 0.65",
        "free_thresh: 0.196",
    )

    for item in required:
        assert item in text


def test_cartographer_pbstream_is_not_runtime_map_asset():
    assert not (
        ASSET_DIR
        / "mayday_supervised_route_03.pbstream"
    ).exists()

"""Versioned confidence-calibration configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class TemperatureScalingConfig:
    """Deterministic bounded optimization settings for temperature scaling."""

    schema_version: str
    method: str
    minimum_temperature: float
    maximum_temperature: float
    optimization_iterations: int
    reliability_bin_count: int

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported calibration configuration schema_version")
        if self.method != "temperature_scaling":
            raise ValueError("method must be temperature_scaling")
        for field_name in ("minimum_temperature", "maximum_temperature"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a real number")
            canonical_value = float(value)
            if not isfinite(canonical_value) or canonical_value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
            object.__setattr__(self, field_name, canonical_value)
        if not self.minimum_temperature < 1.0 < self.maximum_temperature:
            raise ValueError("temperature bounds must contain the identity temperature 1.0")
        for field_name in ("optimization_iterations", "reliability_bin_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
        if self.optimization_iterations < 10:
            raise ValueError("optimization_iterations must be at least 10")
        if not 2 <= self.reliability_bin_count <= 100:
            raise ValueError("reliability_bin_count must be between 2 and 100")


def load_temperature_scaling_config(path: Path) -> TemperatureScalingConfig:
    """Load strict temperature-scaling configuration from TOML."""
    with path.open("rb") as config_file:
        document: dict[str, Any] = tomllib.load(config_file)
    expected_fields = {
        "schema_version",
        "method",
        "minimum_temperature",
        "maximum_temperature",
        "optimization_iterations",
        "reliability_bin_count",
    }
    unexpected_fields = set(document) - expected_fields
    if unexpected_fields:
        raise ValueError(f"unsupported calibration fields: {sorted(unexpected_fields)}")
    missing_fields = expected_fields - set(document)
    if missing_fields:
        raise ValueError(f"missing calibration fields: {sorted(missing_fields)}")
    return TemperatureScalingConfig(
        schema_version=_require_string(document["schema_version"], "schema_version"),
        method=_require_string(document["method"], "method"),
        minimum_temperature=_require_number(
            document["minimum_temperature"], "minimum_temperature"
        ),
        maximum_temperature=_require_number(
            document["maximum_temperature"], "maximum_temperature"
        ),
        optimization_iterations=_require_integer(
            document["optimization_iterations"], "optimization_iterations"
        ),
        reliability_bin_count=_require_integer(
            document["reliability_bin_count"], "reliability_bin_count"
        ),
    )


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    return float(value)


def _require_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    return cast(int, value)

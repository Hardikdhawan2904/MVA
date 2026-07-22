"""app/services/models_registry/algorithm_spec.py — Stage 6 of the Agent 3
redesign (plan "zany-giggling-crayon"): the AlgorithmSpec shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

COST_CHEAP = "cheap"
COST_MODERATE = "moderate"
COST_EXPENSIVE = "expensive"


@dataclass
class AlgorithmRequirements:
    requires_ml: bool
    min_rows: int = 1
    requires_datetime: bool = False
    requires_target: bool = False
    supports_categorical: bool = False


@dataclass
class AlgorithmSpec:
    algorithm: str
    purpose: str                    # the analysis_type this serves
    requirements: AlgorithmRequirements
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    cost_tier: str = COST_CHEAP
    supported_dataset_types: list[str] = field(default_factory=list)
    implementation_class: str = ""
    enabled: bool = True

    @staticmethod
    def from_dict(d: dict) -> "AlgorithmSpec":
        req = d.get("requirements", {})
        return AlgorithmSpec(
            algorithm=d["algorithm"],
            purpose=d["purpose"],
            requirements=AlgorithmRequirements(
                requires_ml=bool(req.get("requires_ml", False)),
                min_rows=int(req.get("min_rows", 1)),
                requires_datetime=bool(req.get("requires_datetime", False)),
                requires_target=bool(req.get("requires_target", False)),
                supports_categorical=bool(req.get("supports_categorical", False)),
            ),
            inputs=list(d.get("inputs", [])),
            outputs=list(d.get("outputs", [])),
            cost_tier=d.get("cost_tier", COST_CHEAP),
            supported_dataset_types=list(d.get("supported_dataset_types", [])),
            implementation_class=d.get("implementation_class", ""),
            enabled=bool(d.get("enabled", True)),
        )

"""
Labeling module for Pipeline V2.
Implements physical elapsed-time 5-second horizon labeling.
"""

from pipeline_v2.labeling.physical_horizon_labeler import (
    LabelingValidationReport,
    PhysicalHorizonLabeler,
)

__all__ = [
    "LabelingValidationReport",
    "PhysicalHorizonLabeler",
]

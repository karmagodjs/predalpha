"""
Ingestion module for Pipeline V2.
Handles raw market event normalization into canonical events.
"""

from pipeline_v2.ingestion.event_normalizer import (
    CanonicalEvent,
    EventNormalizer,
    NormalizationReport,
    RejectReason,
)

__all__ = [
    "CanonicalEvent",
    "EventNormalizer",
    "NormalizationReport",
    "RejectReason",
]

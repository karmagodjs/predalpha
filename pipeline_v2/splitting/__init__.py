"""
Splitting module for Pipeline V2.
Implements strictly chronological, purged temporal train/validation/test splitting.
"""

from pipeline_v2.splitting.temporal_purged_split import (
    calculate_required_purge_ms,
    PurgedTemporalSplitter,
    SplitReport,
    SplitMetadata,
)

__all__ = [
    "calculate_required_purge_ms",
    "PurgedTemporalSplitter",
    "SplitReport",
    "SplitMetadata",
]

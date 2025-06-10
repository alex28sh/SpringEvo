"""SpringEvo extractor package.

High-level helpers are re-exported for convenience so that external
code can simply do `from extractor import take_snapshot, diff_snapshots`.
"""

from .api_extractor import take_snapshot  # noqa: F401
from .diff import diff_snapshots  # noqa: F401 
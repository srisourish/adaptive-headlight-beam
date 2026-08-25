"""
diagnostics/__init__.py — System Health & Predictive Diagnostics package.

This package is intentionally separate from fusion/explainability.py.

  fusion/explainability.py   → explains *driving decisions* (why a zone was dimmed).
  diagnostics/               → explains *pipeline health* (why the system might fail).

Public API
----------
    from diagnostics import HealthMonitor, HealthFinding, HealthSeverity
    from diagnostics import record, get_monitor
"""

from diagnostics.severity import HealthFinding, HealthSeverity, worst_severity
from diagnostics.health_explainer import HealthExplainer
from diagnostics.health_monitor import HealthMonitor, get_monitor, record

__all__ = [
    "HealthMonitor",
    "HealthFinding",
    "HealthSeverity",
    "HealthExplainer",
    "worst_severity",
    "get_monitor",
    "record",
]

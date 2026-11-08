"""Shared infrastructure for the Runtime-Bound experiments.

Experiments import from here and never from each other. Public surface:

    from rbcore import ExperimentConfig, RunContext, load_config
"""

from rbcore.config import ExperimentConfig, load_config
from rbcore.run import RunContext

__all__ = ["ExperimentConfig", "RunContext", "load_config"]
__version__ = "0.1.0"

"""
ROCKETSHIP: A flexible and modular software tool for the planning, 
processing and analysis of dynamic MRI studies.

This is the Python port of the original MATLAB version.
"""

__version__ = "2.0.0"
__author__ = "Thomas Ng, Samuel Barnes"
__email__ = "thomasn@caltech.edu, srbarnes@caltech.edu"

from . import dce
from . import parametric
from . import utils

__all__ = ['dce', 'parametric', 'utils']
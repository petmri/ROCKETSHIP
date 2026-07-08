"""ASCII art banner for the ROCKETSHIP Python CLIs."""

from __future__ import annotations

import sys
from typing import IO

from version import __version__

# The version sits in a fixed-width field so the trailing "|_***" stays aligned
# regardless of the version string's length.
_VERSION_FIELD_WIDTH = 21

_BANNER_TEMPLATE = r"""
                                 /
                                //
                               ///
         _/-------------------////--
     ___/                           \ _   ***
   _/        ROCKETSHIP       _____  | ***
    \___        {version_field}|_***
        \_                          /     ***
          \-------------------\\\\--
                               \\\
                                \\
                                 \
"""


def _render_banner() -> str:
    version_field = f"v{__version__}".ljust(_VERSION_FIELD_WIDTH)
    return _BANNER_TEMPLATE.format(version_field=version_field)


def print_banner(stream: IO[str] | None = None) -> None:
    """Print the ROCKETSHIP banner.

    Defaults to stderr so it does not pollute the JSON event stream on stdout.
    """
    if stream is None:
        stream = sys.stderr
    stream.write(_render_banner() + "\n")
    stream.flush()

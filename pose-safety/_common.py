"""Put ../common — the code shared by the demos in this repo — on sys.path.

Every script in this folder that is run directly starts with ``import _common``,
after which ``camera_settings``, ``hud``, ``web_view``, ``systemdata`` and the
``detectors`` package import as before.
"""

import os
import sys

_COMMON = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
if _COMMON not in sys.path:
    sys.path.insert(1, _COMMON)      # after this folder, before site-packages

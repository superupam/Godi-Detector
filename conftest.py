# Ensure the project root is importable when pytest is run from anywhere,
# so the tests can import utils, scraper and the rest as top-level modules.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

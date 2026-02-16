"""Color War Pong — entry point.

Run from the parent directory:
    python -m ColorWar.color_war

Or directly:
    python color_war.py
"""

import sys
import os

# Ensure the parent directory is on sys.path so relative imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ColorWar.main import main


if __name__ == "__main__":
    main()

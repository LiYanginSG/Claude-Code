import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from agent import config


@pytest.fixture
def cfg():
    return config.load(Path(__file__).resolve().parents[1] / "config.yaml")

"""Dashboard smoke test: the app runs end to end on the committed models without errors."""

import pytest
from streamlit.testing.v1 import AppTest

from src import config

TABS = ["Predict", "Campaign watchlist", "What the model learned", "Model performance", "Data overview"]


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(str(config.ROOT / "app" / "streamlit_app.py"), default_timeout=300).run()


def test_dashboard_renders_without_errors(app):
    assert not app.exception
    assert not app.error
    assert [t.label for t in app.tabs] == TABS

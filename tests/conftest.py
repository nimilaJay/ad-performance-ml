"""Shared fixtures: the committed models, metadata and datasets (tests never retrain anything)."""

import pytest

from src import config, data, explain


@pytest.fixture(scope="session")
def artifacts():
    return explain.load_artifacts()


@pytest.fixture(scope="session")
def models(artifacts):
    return artifacts[0]


@pytest.fixture(scope="session")
def meta(artifacts):
    return artifacts[1]


@pytest.fixture(scope="session")
def dataset(artifacts):
    return artifacts[2]


@pytest.fixture(scope="session")
def panel(meta):
    return data.load(config.ROOT / meta["data_path"], verbose=False)

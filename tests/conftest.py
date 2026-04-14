"""Shared pytest fixtures for Stage 5 safety evaluation."""
from __future__ import annotations

import pytest

from retail_support.config import SupportSettings
from retail_support.runtime import RetailSupportOrchestrator, SupportSession


@pytest.fixture(scope="session")
def orchestrator() -> RetailSupportOrchestrator:
    """Build the retail support orchestrator once per test session."""
    settings = SupportSettings.from_env()
    return RetailSupportOrchestrator(settings=settings)


@pytest.fixture
def session(orchestrator: RetailSupportOrchestrator) -> SupportSession:
    """Fresh conversation session for each test to prevent history bleed."""
    return orchestrator.start_session()

"""Guard tests for AsyncOmniEngine.do_log_stats edge cases."""

import asyncio

import pytest

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.asyncio]


async def test_do_log_stats_noop_when_manager_missing():
    """do_log_stats should silently return when logger_manager is None."""
    engine = AsyncOmniEngine.__new__(AsyncOmniEngine)
    engine.logger_manager = None
    engine.orchestrator_loop = None
    await engine.do_log_stats()  # should not raise


async def test_do_log_stats_noop_when_loop_missing():
    """do_log_stats should silently return when orchestrator_loop is None."""
    engine = AsyncOmniEngine.__new__(AsyncOmniEngine)
    engine.logger_manager = object()  # non-None sentinel
    engine.orchestrator_loop = None
    await engine.do_log_stats()  # should not raise


async def test_do_log_stats_noop_when_loop_closed():
    """do_log_stats should silently return when orchestrator_loop is closed."""
    engine = AsyncOmniEngine.__new__(AsyncOmniEngine)
    engine.logger_manager = object()  # non-None sentinel
    loop = asyncio.new_event_loop()
    loop.close()
    engine.orchestrator_loop = loop
    await engine.do_log_stats()  # should not raise

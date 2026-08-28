# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextlib
import time
from unittest.mock import MagicMock

import pytest

import vllm.envs as envs
from vllm.v1.engine.async_llm import (
    AsyncLLM,
    _EngineProgressState,
    _get_engine_progress_watchdog_config,
    _run_engine_progress_watchdog,
)
from vllm.v1.engine.exceptions import EngineDeadError

pytestmark = pytest.mark.skip_global_cleanup


def test_watchdog_config_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "VLLM_ENGINE_PROGRESS_WATCHDOG", False)
    assert _get_engine_progress_watchdog_config() is None

    monkeypatch.setattr(envs, "VLLM_ENGINE_PROGRESS_WATCHDOG", True)
    monkeypatch.setattr(envs, "VLLM_ENGINE_PROGRESS_CHECK_INTERVAL_S", 10.0)
    monkeypatch.setattr(envs, "VLLM_ENGINE_PROGRESS_TIMEOUT_S", 60.0)
    assert _get_engine_progress_watchdog_config() == (10.0, 60.0)


@pytest.mark.parametrize(
    ("check_interval_s", "timeout_s"),
    [(0.0, 60.0), (10.0, 10.0), (10.0, 5.0)],
)
def test_watchdog_config_rejects_invalid_timing(
    monkeypatch: pytest.MonkeyPatch,
    check_interval_s: float,
    timeout_s: float,
) -> None:
    monkeypatch.setattr(envs, "VLLM_ENGINE_PROGRESS_WATCHDOG", True)
    monkeypatch.setattr(envs, "VLLM_ENGINE_PROGRESS_CHECK_INTERVAL_S", check_interval_s)
    monkeypatch.setattr(envs, "VLLM_ENGINE_PROGRESS_TIMEOUT_S", timeout_s)

    with pytest.raises(ValueError):
        _get_engine_progress_watchdog_config()


@pytest.mark.parametrize(
    ("local_engine_count", "enable_elastic_ep"),
    [(0, False), (2, False), (1, True)],
)
def test_watchdog_config_rejects_unsupported_engine_topology(
    monkeypatch: pytest.MonkeyPatch,
    local_engine_count: int,
    enable_elastic_ep: bool,
) -> None:
    monkeypatch.setattr(envs, "VLLM_ENGINE_PROGRESS_WATCHDOG", True)

    with pytest.raises(ValueError):
        _get_engine_progress_watchdog_config(
            local_engine_count=local_engine_count,
            enable_elastic_ep=enable_elastic_ep,
        )


def test_new_requests_do_not_extend_active_deadline() -> None:
    state = _EngineProgressState()
    state.record_request_started()
    first_request_at = state.last_progress_at

    state.record_request_started()

    assert state.last_progress_at == first_request_at


def test_suspend_and_resume_reset_the_deadline() -> None:
    state = _EngineProgressState()
    state.record_request_started()
    state.last_progress_at = 0.0
    state.suspend()

    state.resume(active=True)

    assert state.suspended is False
    assert state.active is True
    assert state.last_progress_at > 0.0


def test_watchdog_terminates_a_stalled_active_engine() -> None:
    output_processor = MagicMock()
    output_processor.has_unfinished_requests.return_value = True
    output_processor.get_num_unfinished_requests.return_value = 3
    terminate_process = MagicMock()
    state = _EngineProgressState()
    state.record_request_started()
    state.last_progress_at = time.monotonic() - 1.0

    asyncio.run(
        _run_engine_progress_watchdog(
            output_processor,
            state,
            check_interval_s=0,
            timeout_s=0.1,
            terminate_process=terminate_process,
        )
    )

    error = output_processor.propagate_error.call_args.args[0]
    assert isinstance(error, EngineDeadError)
    assert state.timed_out is True
    terminate_process.assert_called_once_with()

    llm = object.__new__(AsyncLLM)
    llm.output_handler = None
    llm._engine_progress_state = state
    assert llm.is_running is False


def test_watchdog_terminates_when_error_propagation_fails() -> None:
    output_processor = MagicMock()
    output_processor.has_unfinished_requests.return_value = True
    output_processor.get_num_unfinished_requests.return_value = 1
    output_processor.propagate_error.side_effect = RuntimeError("queue closed")
    terminate_process = MagicMock()
    state = _EngineProgressState()
    state.record_request_started()
    state.last_progress_at = time.monotonic() - 1.0

    asyncio.run(
        _run_engine_progress_watchdog(
            output_processor,
            state,
            check_interval_s=0,
            timeout_s=0.1,
            terminate_process=terminate_process,
        )
    )

    terminate_process.assert_called_once_with()


@pytest.mark.parametrize("suspended", [False, True])
def test_watchdog_ignores_inactive_or_suspended_engine(suspended: bool) -> None:
    async def run_checks() -> None:
        task = asyncio.create_task(
            _run_engine_progress_watchdog(
                output_processor,
                state,
                check_interval_s=0.001,
                timeout_s=0.01,
                terminate_process=terminate_process,
            )
        )
        await asyncio.sleep(0.005)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    output_processor = MagicMock()
    output_processor.has_unfinished_requests.return_value = suspended
    terminate_process = MagicMock()
    state = _EngineProgressState()
    state.active = suspended
    state.suspended = suspended
    state.last_progress_at = time.monotonic() - 1.0

    asyncio.run(run_checks())

    assert state.timed_out is False
    terminate_process.assert_not_called()

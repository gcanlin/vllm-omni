# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for profiler trace collection across ranks.

Tests that:
- OmniTorchProfilerWrapper writes trace files for each rank
- DiffusionWorker start/stop_profile lifecycle works per rank
- OmniStage handles profiler tasks via inline engine when queues are absent
"""

import os
import tempfile

import pytest
from pytest_mock import MockerFixture
from vllm.config import ProfilerConfig

from vllm_omni.entrypoints.omni_stage import OmniStage
from vllm_omni.entrypoints.stage_utils import OmniStageTaskType
from vllm_omni.profiler.omni_torch_profiler import OmniTorchProfilerWrapper

pytestmark = [pytest.mark.cpu]


# ---------------------------------------------------------------------------
# OmniTorchProfilerWrapper: per-rank trace file naming
# ---------------------------------------------------------------------------


class TestProfilerTraceNaming:
    """Verify that each rank produces a uniquely named trace file."""

    def test_trace_filename_includes_rank(self):
        """_on_trace_ready should produce <filename>_rank<N>.json."""
        with tempfile.TemporaryDirectory() as trace_dir:
            config = ProfilerConfig(
                profiler="torch",
                torch_profiler_dir=trace_dir,
            )
            for rank in (0, 1):
                profiler = OmniTorchProfilerWrapper(
                    profiler_config=config,
                    worker_name=f"test_rank_{rank}",
                    local_rank=rank,
                    activities=["CPU"],
                )
                profiler.set_trace_filename("test_trace")

                # Start → do nothing → stop triggers _on_trace_ready
                profiler.start()
                profiler.stop()

            # Both rank files should exist
            files = sorted(os.listdir(trace_dir))
            rank0_files = [f for f in files if "_rank0.json" in f]
            rank1_files = [f for f in files if "_rank1.json" in f]
            assert rank0_files, f"No rank-0 trace found in {files}"
            assert rank1_files, f"No rank-1 trace found in {files}"

    def test_trace_filename_with_full_path(self):
        """When filename already contains a directory, use as-is."""
        with tempfile.TemporaryDirectory() as trace_dir:
            config = ProfilerConfig(
                profiler="torch",
                torch_profiler_dir=trace_dir,
            )
            profiler = OmniTorchProfilerWrapper(
                profiler_config=config,
                worker_name="test",
                local_rank=3,
                activities=["CPU"],
            )
            full_path = os.path.join(trace_dir, "subdir", "my_trace")
            profiler.set_trace_filename(full_path)
            profiler.start()
            profiler.stop()

            expected = f"{full_path}_rank3.json"
            assert os.path.exists(expected), (
                f"Expected {expected}, found: {os.listdir(os.path.dirname(expected))}"
            )

    def test_get_results_returns_trace_path(self):
        """get_results() should return the path of the exported trace."""
        with tempfile.TemporaryDirectory() as trace_dir:
            config = ProfilerConfig(
                profiler="torch",
                torch_profiler_dir=trace_dir,
                torch_profiler_use_gzip=False,
            )
            profiler = OmniTorchProfilerWrapper(
                profiler_config=config,
                worker_name="test",
                local_rank=0,
                activities=["CPU"],
            )
            profiler.set_trace_filename("results_test")
            profiler.start()
            profiler.stop()

            results = profiler.get_results()
            assert results["trace"] is not None
            assert results["trace"].endswith("_rank0.json")
            assert os.path.exists(results["trace"])


# ---------------------------------------------------------------------------
# DiffusionWorker: profiler lifecycle
# ---------------------------------------------------------------------------


class TestDiffusionWorkerProfiler:
    """Test DiffusionWorker.start_profile / stop_profile."""

    @pytest.fixture
    def worker_with_profiler(self, mocker: MockerFixture):
        """Create a DiffusionWorker with a real profiler (CPU-only)."""
        from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

        config = mocker.Mock()
        config.num_gpus = 1
        config.master_port = 12345
        config.enable_sleep_mode = False
        config.cache_backend = None
        config.cache_config = None
        config.model = "test-model"
        config.profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=tempfile.mkdtemp(),
            torch_profiler_use_gzip=False,
        )

        mocker.patch.object(DiffusionWorker, "init_device")
        mocker.patch.object(DiffusionWorker, "load_model")
        mocker.patch.object(DiffusionWorker, "init_lora_manager")

        worker = DiffusionWorker(
            local_rank=0, rank=0, od_config=config, skip_load_model=True,
        )
        worker.model_runner = mocker.Mock()
        return worker

    def test_start_stop_creates_trace(self, worker_with_profiler):
        """start_profile + stop_profile should produce a trace file."""
        worker = worker_with_profiler
        trace_dir = worker.od_config.profiler_config.torch_profiler_dir

        template = os.path.join(trace_dir, "test_worker")
        worker.start_profile(template)
        worker.stop_profile()

        files = os.listdir(trace_dir)
        assert any("_rank0.json" in f for f in files), f"No rank-0 trace in {files}"

    def test_stop_profile_returns_results(self, worker_with_profiler):
        """stop_profile should return dict with trace path."""
        worker = worker_with_profiler
        trace_dir = worker.od_config.profiler_config.torch_profiler_dir

        template = os.path.join(trace_dir, "test_results")
        worker.start_profile(template)
        result = worker.stop_profile()

        assert isinstance(result, dict)
        assert "trace" in result
        assert result["trace"] is not None
        assert os.path.exists(result["trace"])

    def test_multiple_ranks_produce_separate_traces(self, mocker: MockerFixture):
        """Two workers with different local_rank should write separate files."""
        from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

        trace_dir = tempfile.mkdtemp()

        workers = []
        for rank in (0, 1):
            config = mocker.Mock()
            config.num_gpus = 2
            config.master_port = 12345
            config.enable_sleep_mode = False
            config.cache_backend = None
            config.cache_config = None
            config.model = "test-model"
            config.profiler_config = ProfilerConfig(
                profiler="torch",
                torch_profiler_dir=trace_dir,
                torch_profiler_use_gzip=False,
            )

            mocker.patch.object(DiffusionWorker, "init_device")
            mocker.patch.object(DiffusionWorker, "load_model")
            mocker.patch.object(DiffusionWorker, "init_lora_manager")

            worker = DiffusionWorker(
                local_rank=rank, rank=rank, od_config=config, skip_load_model=True,
            )
            worker.model_runner = mocker.Mock()
            workers.append(worker)

        # Start and stop profiling on both workers
        template = os.path.join(trace_dir, "multi_rank")
        for w in workers:
            w.start_profile(template)
        for w in workers:
            w.stop_profile()

        files = os.listdir(trace_dir)
        rank0_files = [f for f in files if "_rank0.json" in f]
        rank1_files = [f for f in files if "_rank1.json" in f]
        assert rank0_files, f"Missing rank-0 trace in {files}"
        assert rank1_files, f"Missing rank-1 trace in {files}"


# ---------------------------------------------------------------------------
# OmniStage: inline engine profiler routing
# ---------------------------------------------------------------------------


class TestOmniStageInlineProfiler:
    """Test that OmniStage routes profiler tasks to inline engine."""

    @pytest.fixture
    def stage_with_inline_engine(self, mocker: MockerFixture):
        """Create an OmniStage with a mock inline engine (no queues)."""
        stage_config = mocker.Mock()
        stage_config.stage_id = 0
        stage_config.engine_args = mocker.Mock()
        stage_config.engine_args.model_stage = "diffusion"
        stage_config.engine_args.engine_output_type = None
        stage_config.engine_args.stage_id = 0
        stage_config.runtime = mocker.Mock()
        stage_config.runtime.requires_multimodal_data = False
        stage_config.stage_type = "diffusion"
        stage_config.final_output = True
        stage_config.final_output_type = "video"
        stage_config.is_comprehension = False
        # No custom_process_input_func
        del stage_config.custom_process_input_func
        # No prompt_expand_func
        del stage_config.prompt_expand_func
        # Default sampling params
        stage_config.default_sampling_params = {}
        # No input sources
        stage_config.input_sources = []
        stage_config.engine_input_source = []

        # Patch SamplingParams import to avoid full init
        mocker.patch(
            "vllm_omni.entrypoints.omni_stage.OmniDiffusionSamplingParams",
            return_value=mocker.Mock(),
        )

        stage = OmniStage(stage_config)

        # Attach a mock inline engine (simulates inline diffusion mode)
        mock_engine = mocker.Mock()
        mock_engine.start_profile = mocker.Mock()
        mock_engine.stop_profile = mocker.Mock(return_value={"traces": ["t.json"], "tables": []})
        stage._inline_engine = mock_engine

        return stage, mock_engine

    def test_submit_profiler_start_routes_to_inline_engine(self, stage_with_inline_engine):
        """submit(PROFILER_START) should call inline_engine.start_profile()."""
        stage, mock_engine = stage_with_inline_engine

        stage.submit({"type": OmniStageTaskType.PROFILER_START})

        mock_engine.start_profile.assert_called_once()

    def test_submit_profiler_stop_routes_to_inline_engine(self, stage_with_inline_engine):
        """submit(PROFILER_STOP) should call inline_engine.stop_profile()."""
        stage, mock_engine = stage_with_inline_engine

        stage.submit({"type": OmniStageTaskType.PROFILER_STOP})

        mock_engine.stop_profile.assert_called_once()

    def test_stop_profile_returns_inline_engine_result(self, stage_with_inline_engine):
        """stop_profile() should return the inline engine's result directly."""
        stage, mock_engine = stage_with_inline_engine

        result = stage.stop_profile()

        mock_engine.stop_profile.assert_called_once()
        assert result == {"traces": ["t.json"], "tables": []}

    def test_submit_asserts_when_no_queue_and_no_inline_engine(self, mocker: MockerFixture):
        """submit() should assert when neither queues nor inline engine available."""
        stage_config = mocker.Mock()
        stage_config.stage_id = 0
        stage_config.engine_args = mocker.Mock()
        stage_config.engine_args.model_stage = "diffusion"
        stage_config.engine_args.engine_output_type = None
        stage_config.engine_args.stage_id = 0
        stage_config.runtime = mocker.Mock()
        stage_config.runtime.requires_multimodal_data = False
        stage_config.stage_type = "diffusion"
        stage_config.final_output = False
        stage_config.final_output_type = None
        stage_config.is_comprehension = False
        del stage_config.custom_process_input_func
        del stage_config.prompt_expand_func
        stage_config.default_sampling_params = {}
        stage_config.input_sources = []
        stage_config.engine_input_source = []

        mocker.patch(
            "vllm_omni.entrypoints.omni_stage.OmniDiffusionSamplingParams",
            return_value=mocker.Mock(),
        )

        stage = OmniStage(stage_config)
        # No inline engine, no queues
        assert stage._inline_engine is None
        assert stage._in_q is None

        with pytest.raises(AssertionError):
            stage.submit({"type": OmniStageTaskType.PROFILER_START})

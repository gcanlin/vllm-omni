# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

StateT = TypeVar("StateT")
OutputT = TypeVar("OutputT")


class StaticGraphBackend(ABC):
    """Backend-specific graph capture and replay interface."""

    device_type: str

    @abstractmethod
    def is_available(self, device: torch.device) -> bool:
        """Return whether this backend can capture graphs on the given device."""

    @abstractmethod
    def create_graph(self) -> Any:
        """Create an empty backend graph object."""

    @abstractmethod
    @contextmanager
    def capture_graph(self, graph: Any, pool: Any | None = None):
        """Context manager used to capture a callable into the graph."""

    @abstractmethod
    def replay(self, graph: Any) -> None:
        """Replay a captured graph."""

    @abstractmethod
    def synchronize(self, device: torch.device | None = None) -> None:
        """Synchronize the backend stream before or after capture when needed."""

    def get_graph_pool(self, use_global_graph_pool: bool) -> Any | None:
        """Return a graph pool handle when the backend supports it."""
        if not use_global_graph_pool:
            return None
        try:
            from vllm.platforms import current_platform

            return current_platform.get_global_graph_pool()
        except Exception:
            logger.debug("Static graph backend could not retrieve a global graph pool", exc_info=True)
            return None


class CUDAGraphBackend(StaticGraphBackend):
    device_type = "cuda"

    def is_available(self, device: torch.device) -> bool:
        return device.type == "cuda" and torch.cuda.is_available()

    def create_graph(self) -> Any:
        return torch.cuda.CUDAGraph()

    @contextmanager
    def capture_graph(self, graph: Any, pool: Any | None = None):
        with torch.cuda.graph(graph, pool=pool):
            yield

    def replay(self, graph: Any) -> None:
        graph.replay()

    def synchronize(self, device: torch.device | None = None) -> None:
        torch.cuda.synchronize(device)


class NPUGraphBackend(StaticGraphBackend):
    device_type = "npu"

    def is_available(self, device: torch.device) -> bool:
        return (
            device.type == "npu"
            and hasattr(torch, "npu")
            and hasattr(torch.npu, "is_available")
            and torch.npu.is_available()
        )

    def create_graph(self) -> Any:
        return torch.npu.NPUGraph()

    @contextmanager
    def capture_graph(self, graph: Any, pool: Any | None = None):
        with torch.npu.graph(graph, pool=pool):
            yield

    def replay(self, graph: Any) -> None:
        graph.replay()

    def synchronize(self, device: torch.device | None = None) -> None:
        # torch.npu.synchronize() accepts no device argument.
        torch.npu.synchronize()


class BucketPolicy(ABC):
    """Maps runtime sizes to capture buckets."""

    @property
    @abstractmethod
    def capture_sizes(self) -> list[int]:
        """All capture sizes that should be warmed up and captured."""

    @abstractmethod
    def get_bucket(self, actual_size: int) -> int | None:
        """Return the smallest captured size that can serve the runtime input."""


class ListBucketPolicy(BucketPolicy):
    """Use a fixed ordered list of bucket sizes."""

    def __init__(self, capture_sizes: Iterable[int]):
        self._capture_sizes = sorted({int(x) for x in capture_sizes if int(x) > 0})

    @property
    def capture_sizes(self) -> list[int]:
        return self._capture_sizes

    def get_bucket(self, actual_size: int) -> int | None:
        for size in self._capture_sizes:
            if actual_size <= size:
                return size
        return None


class PowerOfTwoBucketPolicy(ListBucketPolicy):
    """Capture power-of-two batch sizes up to a configured maximum."""

    @classmethod
    def from_max_batch_size(cls, max_batch_size: int) -> "PowerOfTwoBucketPolicy":
        bucket_sizes = [1 << i for i in range(max_batch_size.bit_length()) if (1 << i) <= max_batch_size]
        if max_batch_size not in bucket_sizes:
            bucket_sizes.append(max_batch_size)
        return cls(bucket_sizes)


class GraphWorkloadAdapter(ABC, Generic[StateT, OutputT]):
    """Workload-specific logic for graphable subgraphs."""

    def can_run_graph(self, *args: Any, **kwargs: Any) -> bool:
        return True

    @abstractmethod
    def bucket_value(self, *args: Any, **kwargs: Any) -> int:
        """Extract the runtime value used to select a bucket."""

    @abstractmethod
    def build_static_state(self, bucket_size: int, device: torch.device, dtype: torch.dtype) -> StateT:
        """Allocate or retrieve static state for a bucket."""

    @abstractmethod
    def warmup(self, state: StateT) -> None:
        """Run any eager warmup needed before capture."""

    @abstractmethod
    def run(self, state: StateT) -> OutputT:
        """Execute the graphable callable using only static state."""

    @abstractmethod
    def eager(self, *args: Any, **kwargs: Any) -> OutputT:
        """Fallback eager execution."""

    @abstractmethod
    def copy_inputs(self, state: StateT, *args: Any, **kwargs: Any) -> None:
        """Copy runtime inputs into static state before replay."""

    def process_output(self, output: OutputT, state: StateT, *args: Any, **kwargs: Any) -> OutputT:
        """Slice or clone outputs after replay."""
        return output


@dataclass
class CapturedStaticGraph(Generic[StateT, OutputT]):
    graph: Any
    state: StateT
    output: OutputT


class BucketedStaticGraphRunner(Generic[StateT, OutputT]):
    """Reusable bucketed capture/replay runtime for model-local static graphs."""

    def __init__(
        self,
        *,
        backend: StaticGraphBackend,
        bucket_policy: BucketPolicy,
        workload: GraphWorkloadAdapter[StateT, OutputT],
        enabled: bool = True,
        use_global_graph_pool: bool = False,
    ) -> None:
        self.backend = backend
        self.bucket_policy = bucket_policy
        self.workload = workload
        self.enabled = enabled
        self.use_global_graph_pool = use_global_graph_pool

        self._warmed_up = False
        self._graph_pool: Any | None = None
        self._captures: dict[int, CapturedStaticGraph[StateT, OutputT]] = {}

    @property
    def capture_sizes(self) -> list[int]:
        return self.bucket_policy.capture_sizes

    @property
    def warmed_up(self) -> bool:
        return self._warmed_up

    @property
    def captures(self) -> dict[int, CapturedStaticGraph[StateT, OutputT]]:
        return self._captures

    def warmup(
        self,
        device: torch.device,
        dtype: torch.dtype,
        *,
        continue_on_capture_failure: bool = True,
    ) -> None:
        if self._warmed_up or not self.enabled or not self.backend.is_available(device):
            return

        self._graph_pool = self.backend.get_graph_pool(self.use_global_graph_pool)

        for bucket_size in self.bucket_policy.capture_sizes:
            try:
                state = self.workload.build_static_state(bucket_size, device, dtype)
                with torch.no_grad():
                    self.workload.warmup(state)
                self.backend.synchronize(device)

                graph = self.backend.create_graph()
                with torch.no_grad():
                    with self.backend.capture_graph(graph, pool=self._graph_pool):
                        output = self.workload.run(state)

                self._captures[bucket_size] = CapturedStaticGraph(
                    graph=graph,
                    state=state,
                    output=output,
                )
            except Exception:
                logger.warning("Failed to capture static graph for bucket=%s", bucket_size, exc_info=True)
                if not continue_on_capture_failure:
                    raise

        self._warmed_up = True

    def run(self, *args: Any, **kwargs: Any) -> OutputT:
        if not self.enabled or not self._warmed_up or not self.workload.can_run_graph(*args, **kwargs):
            return self.workload.eager(*args, **kwargs)

        bucket = self.bucket_policy.get_bucket(self.workload.bucket_value(*args, **kwargs))
        if bucket is None:
            return self.workload.eager(*args, **kwargs)

        capture = self._captures.get(bucket)
        if capture is None:
            return self.workload.eager(*args, **kwargs)

        self.workload.copy_inputs(capture.state, *args, **kwargs)
        self.backend.replay(capture.graph)
        # Synchronize after replay to ensure proper RNG state for subsequent
        # eager operations like torch.multinomial that use random sampling.
        self.backend.synchronize()
        return self.workload.process_output(capture.output, capture.state, *args, **kwargs)

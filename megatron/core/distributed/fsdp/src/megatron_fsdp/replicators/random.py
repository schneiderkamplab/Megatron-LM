# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from collections import OrderedDict, deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import math

import torch
import torch.distributed as dist

from .base import BucketReplicator, DeltaBufferManager

__all__ = ["RandomBucketReplicator"]

# The step tag is sent as base-16 digits (plus their squares) appended to the
# payload. Small integers survive an fp32 AVG all-reduce exactly, and
# mean(d^2) == mean(d)^2 only holds if every rank sent the same digit, so the
# tag also detects ranks that disagree about which step they are on.
_TAG_BASE = 16
_TAG_DIGITS = 6  # Steps are tagged modulo 16**6 = 2**24.
_TAG_NUMEL = 2 * _TAG_DIGITS
# Oversampling margin, in standard deviations of the selected count, so that a
# draw almost always yields at least the fixed payload length.
_OVERSAMPLE_SIGMAS = 6.0


class RandomBucketReplicator(BucketReplicator):
    """Random sampling replication with step-tagged asynchronous delivery.

    Each step, every bucket sends exactly ``int(compression_rate * numel)``
    elements of its delta buffer, so every rank's payload has the same length
    even if ranks disagree about the step. Elements are drawn with a slightly
    higher probability than ``compression_rate`` and the draw is cut to that
    length, starting at a random offset so the cut-off part varies from step to
    step (in the rare case of too few draws, the payload is zero-padded). The
    selection is a pure function of (seed, bucket_id, step), so all ranks pick
    the same elements without sharing RNG state. The selected values, followed
    by a step tag, are averaged across nodes with an async all_reduce.

    Results are applied once their collective has completed, which may be
    several steps after they were sent. Each rank keeps the indices of its last
    ``max_delay`` steps per bucket and uses them to place a result; a result
    whose indices have already been evicted, or whose tag does not match the
    step it was sent for, is dropped. With ``max_delay=0`` the replicator is
    synchronous: it waits for each step's all_reduce and applies the result in
    the same step, as the reference DeToNATION replicators do.
    """

    def __init__(
        self,
        compression_decay: float = 0.999,
        compression_rate: float = 0.1,
        seed: int = 42,
        max_delay: int = 4,
    ):
        if compression_rate <= 0:
            raise ValueError("compression_rate must be positive")
        if compression_rate > 1:
            raise ValueError("compression_rate > 1.0 is not supported")
        if max_delay < 0:
            raise ValueError("max_delay must be non-negative")

        self.compression_decay = compression_decay
        self.compression_rate = compression_rate
        self.seed = seed
        self.max_delay = max_delay

    def init(
        self,
        process_group: dist.ProcessGroup,
        bucket_sizes: Dict[int, int],
        dtype: torch.dtype,
        device: torch.device,
        lr_provider: Callable[[], float],
    ) -> None:
        self._process_group = process_group
        self._world_size = dist.get_world_size(process_group)
        self._lr_provider = lr_provider
        self._device = device

        self._delta_manager = DeltaBufferManager(bucket_sizes, dtype, device)
        self._bucket_sizes = bucket_sizes
        self._rng = torch.Generator(device=device)

        # Per-bucket step counter, incremented on every replicate call.
        self._steps: Dict[int, int] = {bucket_id: 0 for bucket_id in bucket_sizes}
        # Per-bucket ring buffer of recent selections: step -> indices.
        self._recent_indices: Dict[int, "OrderedDict[int, torch.Tensor]"] = {
            bucket_id: OrderedDict() for bucket_id in bucket_sizes
        }
        # Per-bucket in-flight collectives, oldest first: (step, handle, payload).
        self._pending: Dict[int, Deque[Tuple[int, dist.Work, torch.Tensor]]] = {
            bucket_id: deque() for bucket_id in bucket_sizes
        }

    def pre_step(self) -> None:
        pass

    def post_step(self) -> None:
        pass

    def wait_pending(self, bucket_id: int) -> bool:
        return len(self._pending.get(bucket_id, ())) > 0

    def _select_indices(self, bucket_id: int, step: int, numel: int) -> Tuple[torch.Tensor, int]:
        """Select int(compression_rate * numel) random elements, deterministically from
        (seed, bucket, step).

        Returns (indices, num_valid). Indices past num_valid are padding (a repeat of
        a valid index, or 0) whose payload values must be zeroed. Indices are int32
        when possible to halve the ring buffer's memory.
        """
        num_selected = max(1, int(self.compression_rate * numel))
        rate = self.compression_rate
        oversampled_rate = min(
            1.0, rate + _OVERSAMPLE_SIGMAS * math.sqrt(rate * (1.0 - rate) / numel)
        )
        self._rng.manual_seed(
            ((self.seed * 1_000_003 + bucket_id) * 1_000_003 + step) % (2**63)
        )
        mask = torch.rand(numel, generator=self._rng, device=self._device) < oversampled_rate
        drawn = mask.nonzero().squeeze(1)
        num_drawn = drawn.numel()
        index_dtype = torch.int32 if numel < 2**31 else torch.int64
        if num_drawn >= num_selected:
            # Take num_selected consecutive draws (cyclically) from a random offset.
            offset = torch.randint(
                num_drawn, (1,), generator=self._rng, device=self._device
            )
            positions = (offset + torch.arange(num_selected, device=self._device)) % num_drawn
            return drawn[positions].to(index_dtype), num_selected
        padding_index = drawn[:1] if num_drawn > 0 else torch.zeros(1, dtype=drawn.dtype,
                                                                     device=self._device)
        padded = torch.cat([drawn, padding_index.expand(num_selected - num_drawn)])
        return padded.to(index_dtype), num_drawn

    def _encode_tag(self, step: int) -> torch.Tensor:
        digits = []
        value = step % (_TAG_BASE**_TAG_DIGITS)
        for _ in range(_TAG_DIGITS):
            digits.append(value % _TAG_BASE)
            value //= _TAG_BASE
        digits += [d * d for d in digits]
        return torch.tensor(digits, dtype=torch.float32, device=self._device)

    def _apply_result(
        self, bucket_id: int, step: int, payload: torch.Tensor, grad: torch.Tensor
    ) -> None:
        """Scatter-add a completed result into grad, if its indices are still known."""
        indices = self._recent_indices[bucket_id].get(step)
        if indices is None:
            # Arrived after its indices were evicted from the ring buffer: drop it.
            return
        values = payload[:-_TAG_NUMEL]
        # Checked on device to avoid a host sync; a mismatch zeroes the values.
        tag_ok = (payload[-_TAG_NUMEL:] == self._encode_tag(step)).all()
        values = torch.where(tag_ok, values, torch.zeros_like(values))
        grad.index_add_(0, indices.long(), values.to(grad.dtype))

    def _resolve_completed(self, bucket_id: int, step: int, grad: torch.Tensor) -> None:
        """Apply every in-flight result for this bucket whose collective has completed.

        Collectives on one process group complete in issue order, so polling
        stops at the first one still running. A result older than max_delay
        steps is waited on to bound the number in flight; its indices are gone
        by then, so it is dropped.
        """
        pending = self._pending[bucket_id]
        while pending:
            sent_step, handle, payload = pending[0]
            if step - sent_step > self.max_delay:
                handle.wait()
            elif not handle.is_completed():
                break
            else:
                # Completed: orders the current stream after the collective without
                # blocking the host.
                handle.wait()
            pending.popleft()
            self._apply_result(bucket_id, sent_step, payload, grad)

    def replicate_bucket_group(
        self,
        bucket_group: List[int],
        get_buffer_fn: Callable[[int], torch.Tensor],
        stream: torch.cuda.Stream,
    ) -> Optional[torch.cuda.Event]:
        with torch.cuda.stream(stream):
            for bucket_id in bucket_group:
                grad = get_buffer_fn(bucket_id)
                if grad.numel() == 0:
                    continue
                step = self._steps[bucket_id]
                self._steps[bucket_id] = step + 1

                delta = self._delta_manager.update_delta(
                    bucket_id, grad, self._lr_provider(), self.compression_decay
                )

                if self._world_size <= 1 or self.compression_rate >= 1.0:
                    grad.copy_(delta)
                    delta.zero_()
                    continue

                # Remember this step's selection and evict indices older than max_delay.
                indices, num_valid = self._select_indices(bucket_id, step, delta.numel())
                recent = self._recent_indices[bucket_id]
                recent[step] = indices
                indices = indices.long()
                while len(recent) > self.max_delay + 1:
                    recent.popitem(last=False)

                # The gradient buffer becomes the sum of all results that arrived.
                grad.zero_()
                self._resolve_completed(bucket_id, step, grad)

                # Send this step's selection and remove it from the delta.
                values = delta.index_select(0, indices).to(torch.float32)
                values[num_valid:] = 0  # Padding must not add anything when applied.
                payload = torch.cat([values, self._encode_tag(step)])
                delta.index_fill_(0, indices, 0)
                handle = dist.all_reduce(
                    payload, op=dist.ReduceOp.AVG, group=self._process_group, async_op=True
                )
                if self.max_delay == 0:
                    handle.wait()
                    self._apply_result(bucket_id, step, payload, grad)
                else:
                    self._pending[bucket_id].append((step, handle, payload))

            return stream.record_event()

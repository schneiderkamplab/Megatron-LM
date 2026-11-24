"""Per-step communication accounting for train_lm.py (--profile-steps).

Two independent measurements over a window of training steps:

* Bytes: every torch.distributed collective is wrapped and its payload is
  recorded, labelled by process group as intra-node (FSDP sharding, within a
  node) or inter-node (replication across nodes). Wire bytes per rank are
  estimated with the ring algorithm's send volume.
* Time: a torch.profiler trace of the same window. Collective kernels are
  identified by name; their total time, and the part of it not overlapped by
  any compute kernel ("exposed"), are reported per step.
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist

_WRAPPED = (
    "all_reduce",
    "all_gather",
    "all_gather_into_tensor",
    "reduce_scatter_tensor",
    "broadcast",
)
_COMM_KERNEL_MARKERS = ("nccl", "rccl")


def _numel_bytes(t) -> int:
    if isinstance(t, (list, tuple)):
        return sum(_numel_bytes(x) for x in t)
    return t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0


def _wire_bytes(op: str, args, kwargs, n: int) -> tuple[int, int]:
    """(payload bytes, estimated bytes sent per rank by a ring algorithm)."""
    if op == "all_reduce":
        size = _numel_bytes(args[0] if args else kwargs["tensor"])
        return size, int(2 * (n - 1) / n * size)
    if op == "all_gather":  # (tensor_list, tensor): sends own tensor to n - 1 peers
        size = _numel_bytes(args[1] if len(args) > 1 else kwargs["tensor"])
        return size, (n - 1) * size
    if op == "all_gather_into_tensor":  # (output, input)
        size = _numel_bytes(args[1] if len(args) > 1 else kwargs["input_tensor"])
        return size, (n - 1) * size
    if op == "reduce_scatter_tensor":  # (output, input)
        size = _numel_bytes(args[1] if len(args) > 1 else kwargs["input"])
        return size, int((n - 1) / n * size)
    if op == "broadcast":
        size = _numel_bytes(args[0] if args else kwargs["tensor"])
        return size, size
    return 0, 0


class CommCounter:
    """Wraps torch.distributed collectives and tallies bytes per (group label, op)."""

    def __init__(self, inter_node_groups):
        self.inter_node = {id(g) for g in inter_node_groups}
        self.totals = defaultdict(lambda: {"calls": 0, "payload_bytes": 0, "wire_bytes": 0})
        self._originals = {}
        self.active = False

    def _label(self, group) -> str:
        if group is None or group is dist.group.WORLD:
            return "world"
        return "inter_node" if id(group) in self.inter_node else "intra_node"

    def install(self):
        for name in _WRAPPED:
            original = getattr(dist, name)
            self._originals[name] = original

            def wrapper(*args, _name=name, _original=original, **kwargs):
                if self.active:
                    group = kwargs.get("group")
                    n = dist.get_world_size(group)
                    payload, wire = _wire_bytes(_name, args, kwargs, n)
                    entry = self.totals[(self._label(group), _name)]
                    entry["calls"] += 1
                    entry["payload_bytes"] += payload
                    entry["wire_bytes"] += wire
                return _original(*args, **kwargs)

            setattr(dist, name, wrapper)

    def uninstall(self):
        for name, original in self._originals.items():
            setattr(dist, name, original)


def _union_length(intervals) -> float:
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        if stop <= end:
            continue
        total += stop - max(start, end)
        end = stop
    return total


def _exposed(spans, compute_spans) -> float:
    """Time covered by `spans` while no compute kernel runs."""
    return _union_length(spans + compute_spans) - _union_length(compute_spans)


def kernel_time_summary(trace_path: Path) -> dict:
    """Collective-kernel time (us) from a chrome trace, in total and per GPU stream.

    Each communicator launches its kernels on its own stream, so the per-stream
    split separates intra-node from inter-node collectives; the kernel names
    listed for each stream identify which is which.
    """
    opener = gzip.open if trace_path.suffix == ".gz" else open
    with opener(trace_path, "rt") as f:
        events = json.load(f)["traceEvents"]
    comm, compute = [], []
    streams = defaultdict(lambda: {"spans": [], "kernels": defaultdict(int)})
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_op"):
            continue
        span = (e["ts"], e["ts"] + e.get("dur", 0))
        name = e.get("name", "")
        if any(m in name.lower() for m in _COMM_KERNEL_MARKERS):
            comm.append(span)
            stream = streams[str(e.get("args", {}).get("stream", e.get("tid")))]
            stream["spans"].append(span)
            stream["kernels"][name.split("(")[0][:60]] += 1
        else:
            compute.append(span)
    window = (max(s for _, s in comm + compute) - min(s for s, _ in comm + compute)) if comm else 0
    return {
        "comm_kernel_us": sum(stop - start for start, stop in comm),
        "comm_busy_us": _union_length(comm),
        # Exposed = time some collective runs while no compute kernel runs.
        "comm_exposed_us": _exposed(comm, compute),
        "gpu_window_us": window,
        "num_comm_kernels": len(comm),
        "per_stream": {
            sid: {
                "busy_us": _union_length(st["spans"]),
                "exposed_us": _exposed(st["spans"], compute),
                "kernels": dict(st["kernels"]),
            }
            for sid, st in streams.items()
        },
    }


class StepProfiler:
    """Profiles steps [start, stop) on this rank and writes a JSON summary."""

    def __init__(self, start: int, stop: int, out_dir: str, inter_node_groups, rank: int):
        self.start, self.stop = start, stop
        self.out_dir = Path(out_dir)
        self.rank = rank
        self.counter = CommCounter(inter_node_groups)
        self.counter.install()
        self.profiler = None
        self.wall_start = None

    def step_begin(self, step: int):
        if step == self.start:
            import time

            torch.cuda.synchronize()
            self.counter.active = True
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
            )
            self.profiler.__enter__()
            self.wall_start = time.time()

    def step_end(self, step: int, extra: dict):
        if step != self.stop - 1 or self.profiler is None:
            return
        import time

        torch.cuda.synchronize()
        wall = time.time() - self.wall_start
        self.profiler.__exit__(None, None, None)
        self.counter.active = False
        self.counter.uninstall()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        trace = self.out_dir / f"trace_rank{self.rank}.json.gz"
        self.profiler.export_chrome_trace(str(trace))
        steps = self.stop - self.start
        bytes_per_step = {
            f"{label}/{op}": {k: v / steps for k, v in entry.items()}
            for (label, op), entry in sorted(self.counter.totals.items())
        }
        kernels = kernel_time_summary(trace)
        for key, value in list(kernels.items()):
            if key == "per_stream":
                for st in value.values():
                    st["busy_us"] /= steps
                    st["exposed_us"] /= steps
                    st["kernels"] = {k: n / steps for k, n in st["kernels"].items()}
            else:
                kernels[key] = value / steps
        summary = {
            **extra,
            "rank": self.rank,
            "profiled_steps": [self.start, self.stop],
            "wall_sec_per_step": wall / steps,
            "bytes_per_step": bytes_per_step,
            "kernel_us_per_step": kernels,
        }
        with open(self.out_dir / f"comm_rank{self.rank}.json", "w") as f:
            json.dump(summary, f, indent=2)
        self.profiler = None

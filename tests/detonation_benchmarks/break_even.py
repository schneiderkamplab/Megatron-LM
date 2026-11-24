"""Break-even analysis: below what inter-node bandwidth does a replication
strategy reach a target loss faster than plain hybrid FSDP ('none')?

Inputs:
  curves.json  validation-loss curve and median step time of each run (wandb export)
  profiles/    comm_profile.py summaries, profiles/<nodes>n/<run>/comm_rank0.json

From the profile of each run on LUMI: step time W, inter-node bytes V sent
per rank, and the inter-node collective time E not overlapped by compute.
Collective kernel time is not used for scaling: it includes waiting for the
slowest peer, which dominates small and asynchronous collectives. Instead the
transfer time at per-rank bandwidth B is V / B, with LUMI's effective
bandwidth B_eff calibrated from the 'none' run at the same node count
(its all-reduce bytes over its collective time). The step grows only by what
no longer fits in the overlap window O:

    T(B) = (W - E) + max(0, V / B - O)

Overlap scenarios:
  measured  sync runs hide what they hid on LUMI (O = V / B_eff - E, >= 0);
            async runs can hide behind a whole step (O = W - E)
  ideal     every run can hide behind a whole step (O = W - E)

Optionally ("sync-free"), the extra intra-node stall that host synchronization
causes in some replicators (intra-node exposed time above 'none') is removed
from W, to show what a host-sync-free implementation would gain.

Time to a target loss = steps to reach it (from the curve) x T(B).

    python break_even.py curves.json profiles/ [--out break_even.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

B_LUMI = 12.9e9  # Measured inter-node all_reduce bus bandwidth per rank on LUMI (bytes/s).
GPUS_PER_NODE = 8
TARGETS = {8: [5.0, 4.6, 4.4], 32: [6.0, 5.6]}


def is_async(run: dict) -> bool:
    return run["strategy"] in ("full", "demo", "random") and run.get("max_delay") != 0


def steps_to_loss(val: list, target: float) -> float | None:
    """First (linearly interpolated) step at which validation loss reaches target."""
    val = sorted(val)
    for (s0, l0), (s1, l1) in zip(val, val[1:]):
        if l1 <= target:
            return s0 + (s1 - s0) * (l0 - target) / (l0 - l1) if l0 > target else s0
    return None


def inter_node_stream(profile: dict) -> dict:
    """The GPU stream of the inter-node communicator, matched by kernels per step."""
    calls = sum(v["calls"] for k, v in profile["bytes_per_step"].items()
                if k.startswith("inter_node/"))
    streams = profile["kernel_us_per_step"]["per_stream"].values()
    return min(streams, key=lambda st: abs(sum(st["kernels"].values()) - calls))


def comm_figures(profile: dict) -> dict:
    """Per-step communication summary of one profiled run (bytes, seconds)."""
    b = profile["bytes_per_step"]
    inter = inter_node_stream(profile)
    intra = [st for st in profile["kernel_us_per_step"]["per_stream"].values() if st is not inter]
    return {
        "step_sec": profile["wall_sec_per_step"],
        "inter_bytes": sum(v["wire_bytes"] for k, v in b.items() if k.startswith("inter_node/")),
        "intra_bytes": sum(v["wire_bytes"] for k, v in b.items() if k.startswith("intra_node/")),
        "inter_busy_sec": inter["busy_us"] / 1e6,
        "inter_exposed_sec": inter["exposed_us"] / 1e6,
        "intra_busy_sec": sum(st["busy_us"] for st in intra) / 1e6,
        "intra_exposed_sec": sum(st["exposed_us"] for st in intra) / 1e6,
    }


SCENARIOS = ("measured", "ideal", "measured_syncfree")


def step_time(comm: dict, asynchronous: bool, bandwidths: np.ndarray, b_eff: float,
              ideal: bool, stall: float) -> np.ndarray:
    v, e = comm["inter_bytes"], comm["inter_exposed_sec"]
    base = comm["step_sec"] - e - stall
    overlap = base if (asynchronous or ideal) else max(0.0, v / b_eff - e)
    return base + np.maximum(0.0, v / bandwidths - overlap)


def analyse(runs: dict, profiles: dict, bandwidths: np.ndarray) -> dict:
    out = {}
    b_eff, intra_ref = {}, {}
    for key, profile in profiles.items():
        if key.endswith("/none"):
            c = comm_figures(profile)
            b_eff[key.split("/")[0]] = c["inter_bytes"] / c["inter_busy_sec"]
            intra_ref[key.split("/")[0]] = c["intra_exposed_sec"]
    for key, run in runs.items():
        if key not in profiles:
            continue
        comm = comm_figures(profiles[key])
        nodes_key = key.split("/")[0]
        stall = max(0.0, comm["intra_exposed_sec"] - intra_ref[nodes_key])
        entry = {"comm": comm, "async": is_async(run), "b_eff": b_eff[nodes_key],
                 "host_sync_stall_sec": stall, "targets": {}}
        for scenario in SCENARIOS:
            entry[f"step_sec_{scenario}"] = step_time(
                comm, is_async(run), bandwidths, b_eff[nodes_key],
                ideal=scenario == "ideal",
                stall=stall if scenario.endswith("syncfree") else 0.0).tolist()
        for target in TARGETS[run["nodes"]]:
            steps = steps_to_loss(run["val"], target)
            entry["targets"][str(target)] = {
                "steps": steps,
                **{f"hours_{s}": None if steps is None else
                   (steps * np.array(entry[f"step_sec_{s}"]) / 3600).tolist()
                   for s in SCENARIOS},
            }
        out[key] = entry
    return out


def crossover(results: dict, nodes: int, target: float, scenario: str,
              bandwidths: np.ndarray) -> dict:
    """Highest per-rank bandwidth at which each run reaches `target` before 'none'."""
    base = results[f"{nodes}n/none"]["targets"][str(target)][f"hours_{scenario}"]
    out = {}
    for key, r in results.items():
        if not key.startswith(f"{nodes}n/") or key.endswith("/none"):
            continue
        hours = r["targets"][str(target)][f"hours_{scenario}"]
        if hours is None:
            out[key] = "never reaches target"
        else:
            wins = [b for b, t, t0 in zip(bandwidths, hours, base) if t < t0]
            out[key] = max(wins) if wins else "always slower"
    return out


def gbit_per_node(per_rank: float) -> float:
    return per_rank * GPUS_PER_NODE * 8 / 1e9


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("curves")
    p.add_argument("profiles")
    p.add_argument("--out", default="break_even.json")
    args = p.parse_args()

    runs = json.loads(Path(args.curves).read_text())
    profiles = {f"{f.parent.parent.name}/{f.parent.name}": json.loads(f.read_text())
                for f in Path(args.profiles).glob("*n/*/comm_rank0.json")}
    bandwidths = np.geomspace(0.002e9, B_LUMI, 600)  # bytes/s per rank
    results = analyse(runs, profiles, bandwidths)

    for nodes in (8, 32):
        keys = sorted((k for k in results if k.startswith(f"{nodes}n/")),
                      key=lambda k: results[k]["comm"]["inter_bytes"])
        if not keys:
            continue
        print(f"\n== {nodes} nodes: communication per step (rank 0, profiled steps) ==")
        print(f"{'run':16} {'step s':>6} {'inter GB':>8} {'busy':>6} {'exposed':>7} "
              f"{'intra GB':>8} {'busy':>6} {'exposed':>7} {'stall':>6}")
        for k in keys:
            c = results[k]["comm"]
            print(f"{k:16} {c['step_sec']:6.2f} {c['inter_bytes'] / 1e9:8.2f} "
                  f"{c['inter_busy_sec']:6.3f} {c['inter_exposed_sec']:7.3f} "
                  f"{c['intra_bytes'] / 1e9:8.1f} {c['intra_busy_sec']:6.3f} "
                  f"{c['intra_exposed_sec']:7.3f} {results[k]['host_sync_stall_sec']:6.3f}")
        print(f"effective LUMI inter-node bandwidth: "
              f"{results[keys[0]]['b_eff'] / 1e9:.1f} GB/s per rank")
        for target in TARGETS[nodes]:
            for scenario in SCENARIOS:
                print(f"\n-- target {target}, {scenario} overlap: strategy beats 'none' below --")
                for k, b in crossover(results, nodes, target, scenario, bandwidths).items():
                    text = b if isinstance(b, str) else (
                        f"{b / 1e9:6.3f} GB/s per rank = {gbit_per_node(b):6.1f} Gbit/s per node")
                    print(f"  {k:16} {text}")

    Path(args.out).write_text(json.dumps({
        "bandwidths_per_rank": bandwidths.tolist(), "targets": TARGETS,
        "b_lumi": B_LUMI, "results": results,
    }))


if __name__ == "__main__":
    main()

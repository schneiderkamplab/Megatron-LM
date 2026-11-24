"""Measure inter-node all_reduce bandwidth per rank on LUMI.

One process per GCD over two or more nodes. Ranks r, r + 8, r + 16, ... form
one inter-node group each, exactly like dp_outer in hybrid sharding, and all
groups all_reduce concurrently. With --nccl-net Socket the RCCL network
transport is forced onto plain TCP (set before initialization), to check how
far that throttles inter-node bandwidth while leaving intra-node xGMI alone.

    sbatch --nodes=2 ... --export=...,SCRIPT=tests/detonation_benchmarks/bench_inter_bw.py,ARGS="--nccl-net Socket"
"""
from __future__ import annotations

import argparse
import os
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mb", type=int, default=256, help="payload per all_reduce (MiB)")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--nccl-net", default=None, help="e.g. Socket to force TCP")
    p.add_argument("--socket-ifname", default=None, help="e.g. hsn0 to use one NIC")
    p.add_argument("--socket-nthreads", default=None, help="NCCL_SOCKET_NTHREADS")
    p.add_argument("--nsocks-perthread", default=None, help="NCCL_NSOCKS_PERTHREAD")
    args = p.parse_args()

    if args.nccl_net:
        os.environ["NCCL_NET"] = args.nccl_net
    if args.socket_ifname:
        os.environ["NCCL_SOCKET_IFNAME"] = args.socket_ifname
    if args.socket_nthreads:
        os.environ["NCCL_SOCKET_NTHREADS"] = args.socket_nthreads
    if args.nsocks_perthread:
        os.environ["NCCL_NSOCKS_PERTHREAD"] = args.nsocks_perthread

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    nodes = world // args.gpus_per_node
    assert nodes >= 2, "need at least two nodes"
    my_group = None
    for i in range(args.gpus_per_node):
        ranks = [i + n * args.gpus_per_node for n in range(nodes)]
        group = dist.new_group(ranks)
        if rank in ranks:
            my_group = group

    numel = args.mb * 2**20 // 4
    tensor = torch.randn(numel, device="cuda")
    for _ in range(3):  # warmup, also creates the communicator
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG, group=my_group)
    torch.cuda.synchronize()
    dist.barrier()

    start = time.perf_counter()
    for _ in range(args.iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG, group=my_group)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    dist.barrier()

    # Ring all_reduce sends 2 (n-1)/n of the payload per rank.
    wire = 2 * (nodes - 1) / nodes * numel * 4 * args.iters
    gbps = wire / elapsed / 1e9
    if rank == 0:
        per_node = gbps * args.gpus_per_node * 8
        print(
            f"NCCL_NET={args.nccl_net or 'default'} payload {args.mb} MiB x {args.iters}: "
            f"{gbps:.2f} GB/s per rank = {per_node:.0f} Gbit/s per node "
            f"({elapsed / args.iters * 1e3:.1f} ms per all_reduce)"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

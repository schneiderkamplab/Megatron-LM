# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import math
from typing import Dict, List, Tuple

import torch

__all__ = ["DCTBufferCompress", "DCTBufferTransform"]


class DCTBufferCompress:
    """Top-k compression for DCT-transformed buffer segments.

    Operates on 1-D contiguous tensors (buffer segments) rather than
    per-parameter tensors. Each buffer is reshaped into chunks of
    size `compression_chunk`, and top-k coefficients are selected
    independently per chunk.
    """

    @staticmethod
    @torch.no_grad()
    def compress(
        x: torch.Tensor,
        topk: int,
        chunk_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress a 1-D DCT-encoded buffer via per-chunk top-k selection.

        Args:
            x: 1-D tensor of DCT coefficients, length must be divisible by
               the chunk size used during encoding.
            topk: Number of top coefficients to retain per chunk.
            chunk_size: Size of each chunk. If 0 or not provided, falls
               back to global top-k across the entire buffer.

        Returns:
            Tuple of (indices, values) where indices are int32 and values
            are the selected coefficients. Both are 1-D tensors with length
            topk * num_chunks (or topk if global).
        """
        numel = x.numel()
        if numel == 0:
            return (
                torch.zeros(0, dtype=torch.int32, device=x.device),
                torch.zeros(0, dtype=x.dtype, device=x.device),
            )

        # Global top-k fallback (legacy behavior).
        if chunk_size is None or chunk_size <= 0:
            topk = min(topk, numel)
            idx = torch.topk(x.abs(), k=topk, sorted=False).indices
            val = x[idx]
            return idx.to(torch.int32), val

        # Per-chunk top-k.
        if numel % chunk_size != 0:
            raise ValueError(
                f"Buffer length {numel} not divisible by chunk size {chunk_size}"
            )
        num_chunks = numel // chunk_size
        topk = min(topk, chunk_size)

        x_2d = x.view(num_chunks, chunk_size)
        idx_local = torch.topk(x_2d.abs(), k=topk, dim=-1, sorted=False).indices
        val = torch.gather(x_2d, dim=-1, index=idx_local)

        # Convert local (per-chunk) indices to global indices.
        offsets = torch.arange(num_chunks, device=x.device).unsqueeze(1) * chunk_size
        idx_global = (idx_local + offsets).view(-1)

        return idx_global.to(torch.int32), val.view(-1)

    @staticmethod
    @torch.no_grad()
    def decompress(
        idx: torch.Tensor,
        val: torch.Tensor,
        numel: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Decompress sparse indices/values back to a dense tensor.

        Args:
            idx: 1-D int64 tensor of indices.
            val: 1-D tensor of values at those indices.
            numel: Original tensor length.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Dense 1-D tensor of length `numel`.
        """
        x = torch.zeros(numel, device=device, dtype=dtype)
        if idx.numel() > 0:
            idx_long = idx.to(torch.int64) if idx.dtype != torch.int64 else idx
            x.scatter_(0, idx_long, val)
        return x

    @staticmethod
    @torch.no_grad()
    def batch_decompress(
        idx_list: List[torch.Tensor],
        val_list: List[torch.Tensor],
        numel: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Decompress multiple nodes' sparse representations and average.

        Args:
            idx_list: List of 1-D index tensors (one per node).
            val_list: List of 1-D value tensors (one per node).
            numel: Original tensor length.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Averaged dense 1-D tensor of length `numel`.
        """
        world_size = len(idx_list)
        x = torch.zeros(numel, device=device, dtype=dtype)
        count = torch.zeros(numel, device=device, dtype=dtype)

        for i in range(world_size):
            idx_i = idx_list[i].to(device=device, dtype=torch.int64)
            val_i = val_list[i].to(device=device, dtype=dtype)
            if idx_i.numel() > 0:
                x.scatter_add_(0, idx_i, val_i)
                ones = torch.ones_like(val_i)
                count.scatter_add_(0, idx_i, ones)

        # Average where count > 0
        mask = count > 0
        x[mask] /= count[mask]
        return x


class DCTBufferTransform:
    """DCT/IDCT transform for contiguous buffer segments.

    Unlike DeToNation's DCTTransform which handles per-parameter shapes,
    this version operates on 1-D buffers of known length. The buffer is
    reshaped into chunks of `compression_chunk` size, and DCT is applied
    per-chunk.

    The chunk size is chosen as the closest divisor of the buffer length
    that is <= compression_chunk.
    """

    def __init__(
        self,
        buffer_sizes: Dict[int, int],
        compression_chunk: int,
        dtype: torch.dtype,
        device: torch.device,
        norm: str = "ortho",
    ):
        """
        Args:
            buffer_sizes: Mapping from bucket_id to buffer numel.
            compression_chunk: Target chunk size for DCT.
            dtype: Data type for basis matrices.
            device: Device for basis matrices.
            norm: DCT normalization mode.
        """
        self.compression_chunk = compression_chunk
        self.norm = norm
        self.chunk_sizes: Dict[int, int] = {}
        self.f_dict: Dict[int, torch.Tensor] = {}
        self.b_dict: Dict[int, torch.Tensor] = {}

        for bucket_id, numel in buffer_sizes.items():
            if numel == 0:
                self.chunk_sizes[bucket_id] = 0
                continue
            chunk = _get_smaller_split(numel, compression_chunk)
            self.chunk_sizes[bucket_id] = chunk
            if chunk not in self.f_dict and chunk > 0:
                I = torch.eye(chunk, dtype=dtype, device=device)
                self.f_dict[chunk] = _dct(I, norm=norm)
                self.b_dict[chunk] = _idct(I, norm=norm)

    @torch.no_grad()
    def encode(self, bucket_id: int, x: torch.Tensor) -> torch.Tensor:
        """Apply DCT to a 1-D buffer.

        Reshapes into (num_chunks, chunk_size) and applies DCT per chunk.

        Args:
            bucket_id: Bucket identifier (used to look up chunk size).
            x: 1-D input tensor.

        Returns:
            DCT-encoded 1-D tensor of the same length.
        """
        chunk = self.chunk_sizes[bucket_id]
        if chunk == 0 or x.numel() == 0:
            return x

        numel = x.numel()
        assert numel % chunk == 0, (
            f"Buffer length {numel} not divisible by chunk size {chunk}"
        )
        basis = self.f_dict[chunk].to(x.device)

        x_2d = x.view(-1, chunk)
        encoded = x_2d @ basis.T
        return encoded.view(-1)

    @torch.no_grad()
    def decode(self, bucket_id: int, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse DCT to a 1-D buffer.

        Args:
            bucket_id: Bucket identifier.
            x: 1-D DCT-encoded tensor.

        Returns:
            Decoded 1-D tensor of the same length.
        """
        chunk = self.chunk_sizes[bucket_id]
        if chunk == 0 or x.numel() == 0:
            return x

        numel = x.numel()
        assert numel % chunk == 0, (
            f"Buffer length {numel} not divisible by chunk size {chunk}"
        )
        basis = self.b_dict[chunk].to(x.device)

        x_2d = x.view(-1, chunk)
        decoded = x_2d @ basis.T
        return decoded.view(-1)


# ---------------------------------------------------------------------------
# DCT/IDCT implementations (from torch-dct, adapted)
# ---------------------------------------------------------------------------


def _dct(x: torch.Tensor, norm: str = None) -> torch.Tensor:
    """Type-II DCT over the last dimension."""
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= math.sqrt(N) * 2
        V[:, 1:] /= math.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)
    return V


def _idct(X: torch.Tensor, norm: str = None) -> torch.Tensor:
    """Type-III DCT (inverse of Type-II) over the last dimension."""
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= math.sqrt(N) * 2
        X_v[:, 1:] *= math.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=N, dim=1)
    result = v.new_zeros(v.shape)
    result[:, ::2] += v[:, : N - (N // 2)]
    result[:, 1::2] += v.flip([1])[:, : N // 2]

    return result.view(*x_shape)


# ---------------------------------------------------------------------------
# Divisor utilities (shared by slicing/striding)
# ---------------------------------------------------------------------------


def _get_prime_divisors(n: int) -> list:
    divisors = []
    while n % 2 == 0:
        divisors.append(2)
        n //= 2
    while n % 3 == 0:
        divisors.append(3)
        n //= 3
    i = 5
    while i * i <= n:
        for k in (i, i + 2):
            while n % k == 0:
                divisors.append(k)
                n //= k
        i += 6
    if n > 1:
        divisors.append(n)
    return divisors


def _get_divisors(n: int) -> list:
    if n == 1:
        return [1]
    if n < 1:
        return []
    prime_factors = _get_prime_divisors(n)
    divisors = [1]
    last_prime = 0
    factor = 0
    slice_len = 0
    for prime in prime_factors:
        if last_prime != prime:
            slice_len = len(divisors)
            factor = prime
        else:
            factor *= prime
        for i in range(slice_len):
            divisors.append(divisors[i] * factor)
        last_prime = prime
    divisors.sort()
    return divisors


def _get_smaller_split(n: int, close_to: int) -> int:
    """Find the largest divisor of n that is <= close_to."""
    all_divisors = _get_divisors(n)
    for ix, val in enumerate(all_divisors):
        if val == close_to:
            return val
        if val > close_to:
            if ix == 0:
                return val
            return all_divisors[ix - 1]
    return n

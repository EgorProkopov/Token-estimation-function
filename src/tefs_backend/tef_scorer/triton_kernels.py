from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def triton_available() -> bool:
    return _TRITON_AVAILABLE


if triton_available():
    @triton.jit
    def scatter_keep_mask_kernel(
        Sorted_idx_ptr,
        Keep_sorted_ptr,
        Out_ptr,
        stride_row,
        stride_col,
        seq_len,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < seq_len
        idx = tl.load(Sorted_idx_ptr + pid * stride_row + offs * stride_col, mask=mask, other=0)
        keep = tl.load(Keep_sorted_ptr + pid * stride_row + offs * stride_col, mask=mask, other=0)
        out_offsets = pid * stride_row + idx * stride_col
        tl.store(Out_ptr + out_offsets, keep, mask=mask)


def _triton_block_size(seq_len: int, max_block: int) -> Optional[int]:
    if not _TRITON_AVAILABLE:
        return None
    block = triton.next_power_of_2(seq_len)
    if block > max_block:
        return None
    return block


def scatter_keep_mask(
    sorted_idx: torch.Tensor,
    keep_sorted: torch.Tensor,
    max_block: int = 2048,
) -> torch.Tensor:
    if (
        not _TRITON_AVAILABLE
        or not sorted_idx.is_cuda
        or sorted_idx.dim() != 2
        or keep_sorted.dim() != 2
    ):
        keep_mask = torch.zeros_like(keep_sorted, dtype=torch.bool)
        keep_mask.scatter_(dim=-1, index=sorted_idx, src=keep_sorted)
        return keep_mask

    seq_len = sorted_idx.size(1)
    block = _triton_block_size(seq_len, max_block)
    if block is None:
        keep_mask = torch.zeros_like(keep_sorted, dtype=torch.bool)
        keep_mask.scatter_(dim=-1, index=sorted_idx, src=keep_sorted)
        return keep_mask

    idx_c = sorted_idx.to(torch.int32).contiguous()
    keep_c = keep_sorted.to(torch.int8).contiguous()
    batch = idx_c.size(0)
    stride_row, stride_col = idx_c.stride()
    out = torch.zeros_like(keep_c)
    scatter_keep_mask_kernel[(batch,)](
        idx_c,
        keep_c,
        out,
        stride_row,
        stride_col,
        seq_len,
        BLOCK=block,
    )
    return out.to(torch.bool)

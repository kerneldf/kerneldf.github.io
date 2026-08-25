"""Optimized TPC-H Q6 core using a fused Triton GPU reduction kernel."""

import os
import tempfile
import importlib.util
import linecache
import types

import torch
import triton
import cudf
import pandas as pd


_KERNEL_SOURCE: str = r'''
import triton
import triton.language as tl


@triton.jit
def _q6_kernel(
    shipdate_ptr,
    discount_ptr,
    quantity_ptr,
    price_ptr,
    thresh_ptr,
    out_ptr,
    date_lo,
    date_hi,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    disc_lo = tl.load(thresh_ptr + 0)
    disc_hi = tl.load(thresh_ptr + 1)
    qty_hi = tl.load(thresh_ptr + 2)

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offs < n_elements

    shipdate = tl.load(shipdate_ptr + offs, mask=in_bounds, other=date_hi)
    discount = tl.load(discount_ptr + offs, mask=in_bounds, other=0.0)
    quantity = tl.load(quantity_ptr + offs, mask=in_bounds, other=qty_hi)

    keep = (
        in_bounds
        & (shipdate >= date_lo)
        & (shipdate < date_hi)
        & (discount >= disc_lo)
        & (discount <= disc_hi)
        & (quantity < qty_hi)
    )

    price = tl.load(price_ptr + offs, mask=keep, other=0.0)
    contrib = tl.where(keep, price * discount, 0.0)
    block_sum = tl.sum(contrib, axis=0)
    tl.atomic_add(out_ptr, block_sum, sem="relaxed")
'''


def _load_kernels() -> types.ModuleType:
    """Write Triton GPU kernels to a real Python file and import them for JIT compilation."""
    tmpdir = tempfile.mkdtemp(prefix="q6_triton_")
    path = os.path.join(tmpdir, "_q6_kernels.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_KERNEL_SOURCE)
    linecache.checkcache(path)
    spec = importlib.util.spec_from_file_location("_q6_kernels", path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to create import spec for Triton kernel module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_K: types.ModuleType = _load_kernels()


def _launch_q6(
    l_shipdate_days: torch.Tensor,
    l_discount: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
    discount_lo: float,
    discount_hi: float,
    quantity_hi: float,
    block_size: int = 1024,
) -> torch.Tensor:
    """
    Run the fused Q6 filter/project/reduce on the GPU with Triton.

    The kernel performs one coalesced scan over the predicate columns, conditionally loads
    l_extendedprice only for passing rows, multiplies by l_discount in registers, and atomically
    accumulates one FP64 partial sum per Triton program. This avoids PyTorch's separate mask,
    gather, multiply, and sum temporaries, which is faster for this memory-bound column scan.
    """
    n_elements = l_extendedprice.numel()
    out = torch.zeros(1, dtype=torch.float64, device=l_extendedprice.device)
    if n_elements == 0:
        return out.squeeze(0)

    thresh = torch.tensor(
        [discount_lo, discount_hi, quantity_hi],
        dtype=torch.float64,
        device=l_extendedprice.device,
    )

    grid = (triton.cdiv(n_elements, block_size),)
    _K._q6_kernel[grid](
        l_shipdate_days,
        l_discount,
        l_quantity,
        l_extendedprice,
        thresh,
        out,
        shipdate_lo_days,
        shipdate_hi_days,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=4,
    )
    return out.squeeze(0)


def _query_core(
    l_shipdate_days: torch.Tensor,
    l_discount: torch.Tensor,
    l_quantity: torch.Tensor,
    l_extendedprice: torch.Tensor,
    shipdate_lo_days: int,
    shipdate_hi_days: int,
    discount_lo: float,
    discount_hi: float,
    quantity_hi: float,
) -> torch.Tensor:
    """
    GPU tensor core for TPC-H Q6 using Triton.

    Inputs are already extracted by run_query from cuDF columns into CUDA tensors. This function
    replaces eager PyTorch masking/gathering with one fused Triton GPU kernel, reducing HBM traffic
    and temporary allocations while preserving FP64 aggregate semantics.
    """
    return _launch_q6(
        l_shipdate_days,
        l_discount,
        l_quantity,
        l_extendedprice,
        shipdate_lo_days,
        shipdate_hi_days,
        discount_lo,
        discount_hi,
        quantity_hi,
    )


def run_query(
    tables: dict[str, cudf.DataFrame],
    shipdate_lo: str = "1994-01-01",
    shipdate_hi: str = "1995-01-01",
    discount_lo: float = 0.05,
    discount_hi: float = 0.07,
    quantity_hi: float = 24.0,
) -> float:
    """
    SQL: SELECT sum(l_extendedprice * l_discount) AS revenue FROM lineitem
         WHERE l_shipdate >= CAST('1994-01-01' AS date) AND l_shipdate < CAST('1995-01-01' AS date)
         AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24;
    Description: Computes the forecast revenue increase from eliminating discounts in a given range for a year.
    Params:
        shipdate_lo (str): inclusive lower bound ship date.
        shipdate_hi (str): exclusive upper bound ship date.
        discount_lo (float): inclusive lower bound discount.
        discount_hi (float): inclusive upper bound discount.
        quantity_hi (float): exclusive upper bound quantity.
    """
    df_lineitem = tables["lineitem"]

    shipdate_lo_days = int(pd.to_datetime(shipdate_lo).timestamp() // 86400)
    shipdate_hi_days = int(pd.to_datetime(shipdate_hi).timestamp() // 86400)

    l_shipdate_days_t = torch.as_tensor(
        ((df_lineitem["l_shipdate"].astype("int64") // 86400).astype("int32")).to_cupy()
    )
    l_discount_t = torch.as_tensor(df_lineitem["l_discount"].to_cupy(), dtype=torch.float64)
    l_quantity_t = torch.as_tensor(df_lineitem["l_quantity"].to_cupy(), dtype=torch.float64)
    l_extendedprice_t = torch.as_tensor(df_lineitem["l_extendedprice"].to_cupy(), dtype=torch.float64)

    res = _query_core(
        l_shipdate_days_t,
        l_discount_t,
        l_quantity_t,
        l_extendedprice_t,
        shipdate_lo_days,
        shipdate_hi_days,
        discount_lo,
        discount_hi,
        quantity_hi,
    )
    return float(res.item())

import os
import tempfile
import hashlib
from typing import Any

import torch
import cudf
import pandas as pd
from torch.utils.cpp_extension import load


_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

#define BLOCK_SIZE 256

__global__ void q6_fused_filter_project_sum_kernel(
    const int32_t* __restrict__ shipdate_ptr,
    const double* __restrict__ discount_ptr,
    const double* __restrict__ quantity_ptr,
    const double* __restrict__ price_ptr,
    int32_t date_lo,
    int32_t date_hi,
    double disc_lo,
    double disc_hi,
    double qty_hi,
    double* __restrict__ out_ptr,
    int64_t n_elements
) {
    __shared__ double block_revenue[BLOCK_SIZE];

    const int tid = threadIdx.x;
    double thread_sum = 0.0;

    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
         idx < n_elements;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int32_t shipdate = shipdate_ptr[idx];

        if ((shipdate >= date_lo) && (shipdate < date_hi)) {
            const double discount = discount_ptr[idx];

            if ((discount >= disc_lo) && (discount <= disc_hi)) {
                const double quantity = quantity_ptr[idx];

                if (quantity < qty_hi) {
                    thread_sum += price_ptr[idx] * discount;
                }
            }
        }
    }

    block_revenue[tid] = thread_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            block_revenue[tid] += block_revenue[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_ptr, block_revenue[0]);
    }
}

static torch::Tensor ensure_cuda_contiguous(torch::Tensor tensor, c10::Device device) {
    if (!tensor.is_cuda() || tensor.device() != device) {
        tensor = tensor.to(device);
    }
    return tensor.contiguous();
}

torch::Tensor q6_launcher(
    torch::Tensor shipdate_days,
    torch::Tensor discount,
    torch::Tensor quantity,
    torch::Tensor extendedprice,
    int32_t date_lo,
    int32_t date_hi,
    double disc_lo,
    double disc_hi,
    double qty_hi
) {
    TORCH_CHECK(shipdate_days.scalar_type() == torch::kInt32, "shipdate_days must be int32 epoch days");
    TORCH_CHECK(discount.scalar_type() == torch::kFloat64, "discount must be float64");
    TORCH_CHECK(quantity.scalar_type() == torch::kFloat64, "quantity must be float64");
    TORCH_CHECK(extendedprice.scalar_type() == torch::kFloat64, "extendedprice must be float64");

    TORCH_CHECK(shipdate_days.numel() == discount.numel(), "shipdate_days and discount length mismatch");
    TORCH_CHECK(shipdate_days.numel() == quantity.numel(), "shipdate_days and quantity length mismatch");
    TORCH_CHECK(shipdate_days.numel() == extendedprice.numel(), "shipdate_days and extendedprice length mismatch");

    if (!extendedprice.is_cuda()) {
        extendedprice = extendedprice.to(torch::kCUDA);
    }

    c10::cuda::CUDAGuard device_guard(extendedprice.device());
    const c10::Device device = extendedprice.device();

    shipdate_days = ensure_cuda_contiguous(shipdate_days, device);
    discount = ensure_cuda_contiguous(discount, device);
    quantity = ensure_cuda_contiguous(quantity, device);
    extendedprice = extendedprice.contiguous();

    const int64_t n = extendedprice.numel();
    auto out = torch::zeros({1}, extendedprice.options());

    if (n == 0) {
        return out;
    }

    int cuda_device = 0;
    cudaGetDevice(&cuda_device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_device);

    const int64_t needed_blocks_64 = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int max_resident_style_blocks = std::max(1, prop.multiProcessorCount * 8);
    const int blocks = static_cast<int>(std::min<int64_t>(needed_blocks_64, max_resident_style_blocks));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    q6_fused_filter_project_sum_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        shipdate_days.data_ptr<int32_t>(),
        discount.data_ptr<double>(),
        quantity.data_ptr<double>(),
        extendedprice.data_ptr<double>(),
        date_lo,
        date_hi,
        disc_lo,
        disc_hi,
        qty_hi,
        out.data_ptr<double>(),
        n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("q6_launcher", &q6_launcher, "TPC-H Q6 fused filter/project/sum CUDA launcher");
}
'''


def _load_cuda() -> Any:
    """
    Compile and load the CUDA extension from an in-module source string.

    The hot path runs in a hand-written CUDA kernel. The source is written to a
    stable temporary build directory so PyTorch's extension cache can avoid
    unnecessary rebuilds. The kernel fuses filter, projection, and aggregation
    into one GPU pass, reducing temporary tensors and HBM traffic versus eager
    PyTorch masking.
    """
    source_hash = hashlib.sha1(_CUDA_SOURCE.encode("utf-8")).hexdigest()[:12]
    root_dir = os.path.join(tempfile.gettempdir(), f"q6_cuda_{source_hash}")
    build_dir = os.path.join(root_dir, "build")
    cu_path = os.path.join(root_dir, "q6_kernel.cu")
    os.makedirs(build_dir, exist_ok=True)

    old_source = None
    if os.path.exists(cu_path):
        with open(cu_path, "r", encoding="utf-8") as f:
            old_source = f.read()
    if old_source != _CUDA_SOURCE:
        with open(cu_path, "w", encoding="utf-8") as f:
            f.write(_CUDA_SOURCE)

    return load(
        name=f"q6_cuda_ext_{source_hash}",
        sources=[cu_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-arch=sm_90"],
        with_cuda=True,
        build_directory=build_dir,
        verbose=False,
    )


_EXT = _load_cuda()


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
) -> torch.Tensor:
    """
    Launch the fused CUDA implementation of TPC-H Q6 revenue.

    cuDF/Python work is kept outside this helper; this helper sends already
    materialized 1D tensors to the CUDA extension. The kernel performs one
    coalesced scan, conditionally loads later columns only after earlier
    predicates pass, reduces within each block in shared memory, and issues one
    global double atomic add per block.
    """
    return _EXT.q6_launcher(
        l_shipdate_days,
        l_discount,
        l_quantity,
        l_extendedprice,
        int(shipdate_lo_days),
        int(shipdate_hi_days),
        float(discount_lo),
        float(discount_hi),
        float(quantity_hi),
    )


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
    GPU tensor hot path for TPC-H Q6.

    Runs the numeric filter/project/aggregate in a single hand-written CUDA
    kernel instead of eager PyTorch boolean masks and indexed temporaries. This
    query is memory-bound, so fusing the scan and using conditional loads cuts
    HBM traffic while preserving float64 aggregation semantics.
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

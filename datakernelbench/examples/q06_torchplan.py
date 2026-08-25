import torch
import cudf
import pandas as pd


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
    mask = (
        (l_shipdate_days >= shipdate_lo_days)
        & (l_shipdate_days < shipdate_hi_days)
        & (l_discount >= discount_lo)
        & (l_discount <= discount_hi)
        & (l_quantity < quantity_hi)
    )
    return torch.sum(l_extendedprice[mask] * l_discount[mask])


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

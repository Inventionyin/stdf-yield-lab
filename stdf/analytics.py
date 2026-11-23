"""Analytics helpers for the STDF dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .parser import StdfData


def _cpk(values: pd.Series, lo: float | None, hi: float | None) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if len(vals) < 2:
        return float("nan")
    sigma = vals.std(ddof=1)
    mean = vals.mean()
    if sigma == 0 or not np.isfinite(sigma):
        return float("inf")
    sides = []
    if hi is not None and np.isfinite(hi):
        sides.append((hi - mean) / (3 * sigma))
    if lo is not None and np.isfinite(lo):
        sides.append((mean - lo) / (3 * sigma))
    return min(sides) if sides else float("nan")


def overview(data: StdfData) -> dict:
    parts = data.parts
    results = data.results
    if parts.empty:
        return {"parts": 0, "passed": 0, "yield": 0.0, "lots": 0, "wafers": 0,
                "tests": 0, "records": 0, "test_time": 0.0, "sites": 0, "bins": 0}
    return {
        "parts": int(len(parts)),
        "passed": int(parts["passed"].sum()),
        "failed": int((~parts["passed"]).sum()),
        "yield": float(parts["passed"].mean() * 100),
        "lots": int(parts["lot_id"].nunique()),
        "wafers": int(parts[["lot_id", "wafer_id"]].drop_duplicates().shape[0]),
        "tests": int(results["test_name"].nunique()) if not results.empty else 0,
        "records": int(len(results)),
        "test_time": float(parts["test_time_s"].mean()) if "test_time_s" in parts else 0.0,
        "sites": int(parts["site"].nunique()) if "site" in parts else 0,
        "bins": int(parts["soft_bin"].nunique()) if "soft_bin" in parts else 0,
    }


def yield_by_lot(parts: pd.DataFrame) -> pd.DataFrame:
    if parts.empty:
        return pd.DataFrame(columns=["lot_id", "parts", "passed", "yield"])
    g = parts.groupby("lot_id", dropna=False).agg(
        parts=("passed", "size"), passed=("passed", "sum")
    ).reset_index()
    g["yield"] = g["passed"] / g["parts"] * 100
    return g.sort_values("lot_id")


def yield_by_wafer(parts: pd.DataFrame) -> pd.DataFrame:
    if parts.empty:
        return pd.DataFrame(columns=["lot_id", "wafer_id", "parts", "passed", "yield"])
    g = parts.groupby(["lot_id", "wafer_id"], dropna=False).agg(
        parts=("passed", "size"), passed=("passed", "sum")
    ).reset_index()
    g["yield"] = g["passed"] / g["parts"] * 100
    return g.sort_values(["lot_id", "wafer_id"])


def bin_pareto(parts: pd.DataFrame, bin_names: dict[int, str]) -> pd.DataFrame:
    if parts.empty:
        return pd.DataFrame(columns=["bin", "bin_name", "count", "pct", "cum_pct", "kind"])
    g = parts.groupby("soft_bin", dropna=False).size().reset_index(name="count")
    g["bin_name"] = g["soft_bin"].map(lambda b: bin_names.get(int(b), f"Bin {int(b)}"))
    g["pct"] = g["count"] / g["count"].sum() * 100
    g = g.sort_values("count", ascending=False).reset_index(drop=True)
    g["cum_pct"] = g["pct"].cumsum()
    g["kind"] = np.where(g["soft_bin"] == 1, "PASS", "FAIL")
    return g


def test_stats(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(
            columns=["test_num", "test_name", "units", "n", "mean", "std", "min",
                     "max", "lo", "hi", "cpk", "fail", "fail_rate"]
        )
    rows = []
    for (num, name), grp in results.groupby(["test_num", "test_name"], dropna=False):
        lo = grp["lo_limit"].dropna().iloc[0] if grp["lo_limit"].notna().any() else None
        hi = grp["hi_limit"].dropna().iloc[0] if grp["hi_limit"].notna().any() else None
        vals = grp["value"]
        rows.append(
            {
                "test_num": int(num),
                "test_name": name,
                "units": grp["units"].iloc[0] if "units" in grp else "",
                "n": len(grp),
                "mean": vals.mean(),
                "std": vals.std(ddof=1),
                "min": vals.min(),
                "max": vals.max(),
                "lo": lo,
                "hi": hi,
                "cpk": _cpk(vals, lo, hi),
                "fail": int(grp["failed"].sum()),
                "fail_rate": float(grp["failed"].mean() * 100),
            }
        )
    stats = pd.DataFrame(rows).sort_values("test_num").reset_index(drop=True)
    return stats


def fail_pareto(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["test_name", "fail_count", "pct", "cum_pct"])
    g = results[results["failed"]].groupby("test_name").size().reset_index(name="fail_count")
    g = g.sort_values("fail_count", ascending=False).reset_index(drop=True)
    total = g["fail_count"].sum()
    g["pct"] = g["fail_count"] / total * 100 if total else 0.0
    g["cum_pct"] = g["pct"].cumsum()
    return g


def wafer_table(parts: pd.DataFrame) -> pd.DataFrame:
    g = yield_by_wafer(parts)
    if g.empty:
        return g
    g["fail"] = g["parts"] - g["passed"]
    return g


def wafer_map_data(parts: pd.DataFrame, lot: str, wafer: str) -> pd.DataFrame:
    sub = parts[(parts["lot_id"] == lot) & (parts["wafer_id"] == wafer)].copy()
    return sub.sort_values("part_uid")


def failing_tests_by_part(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["part_uid", "failed_tests"])
    bad = results[results["failed"]]
    g = bad.groupby("part_uid")["test_name"].apply(lambda s: " / ".join(sorted(set(s)))).reset_index()
    g = g.rename(columns={"test_name": "failed_tests"})
    return g


def test_matrix(results: pd.DataFrame, test_names: list[str] | None = None) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    pivot = results.pivot_table(
        index="part_uid", columns="test_name", values="value", aggfunc="first"
    )
    if test_names:
        cols = [c for c in test_names if c in pivot.columns]
        pivot = pivot[cols]
    return pivot


def correlation_matrix(results: pd.DataFrame, max_tests: int = 12) -> pd.DataFrame:
    pivot = test_matrix(results)
    if pivot.empty:
        return pivot
    # Keep the tests with the most complete data.
    pivot = pivot[pivot.notna().sum().sort_values(ascending=False).head(max_tests).index]
    return pivot.corr()


def lot_trend(results: pd.DataFrame, lots: list[str]) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    stats = results.groupby(["lot_id", "test_name"])["value"].mean().reset_index()
    stats = stats[stats["lot_id"].isin(lots)]
    return stats.pivot(index="test_name", columns="lot_id", values="value")


def filter_parts(
    parts: pd.DataFrame, results: pd.DataFrame, lots: list[str], wafers: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = parts[parts["lot_id"].isin(lots)]
    if wafers:
        p = p[p["wafer_id"].isin(wafers)]
    r = results[results["lot_id"].isin(lots)]
    if wafers:
        r = r[r["wafer_id"].isin(wafers)]
    r = r[r["part_uid"].isin(p["part_uid"])]
    return p, r


def unique_die(parts: pd.DataFrame) -> pd.DataFrame:
    """Keep the last test record for each physical die (lot/wafer/x/y)."""
    if parts.empty:
        return parts
    return (
        parts.sort_values("part_uid")
        .drop_duplicates(subset=["lot_id", "wafer_id", "x", "y"], keep="last")
    )


def die_summary(parts: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """One row per physical die, with failed test names."""
    dies = unique_die(parts)
    bad = failing_tests_by_part(results)
    return dies.merge(bad, on="part_uid", how="left")

"""STDF Yield Lab — a Streamlit dashboard for wafer-sort data."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stdf import StdfData, parse_stdf, parse_stdf_bytes
from stdf.analytics import (
    bin_pareto,
    correlation_matrix,
    die_summary,
    fail_pareto,
    failing_tests_by_part,
    overview,
    test_matrix,
    test_stats,
    unique_die,
    wafer_map_data,
    yield_by_lot,
    yield_by_wafer,
)

ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "data"

ACCENT = "#2ec4b6"
ACCENT_2 = "#5aa9e6"
AMBER = "#f2b45c"
RED = "#ef5b5b"
GREEN = "#35c98f"
PANEL = "#0c1c2b"
GRID = "#1b3347"
TEXT = "#dcebf2"
MUTED = "#8fa9b8"

st.set_page_config(
    page_title="STDF Yield Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# styling
# ---------------------------------------------------------------------------

st.markdown(
    f"""
<style>
  html, body, [class*="css"] {{
    font-family: "Inter", "Microsoft YaHei", "PingFang SC", sans-serif;
  }}
  .stApp {{
    background:
      radial-gradient(1200px 600px at 85% -10%, rgba(46,196,182,.10), transparent 55%),
      radial-gradient(900px 500px at 0% 0%, rgba(90,169,230,.08), transparent 50%),
      #07131e;
  }}
  .block-container {{ padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1500px; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  #MainMenu, footer {{ visibility: hidden; }}
  div[data-testid="stToolbar"], div[data-testid="stDecoration"] {{ display: none; }}

  section[data-testid="stSidebar"] {{
    background: #0a1927;
    border-right: 1px solid {GRID};
  }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}

  .hero {{
    position: relative; overflow: hidden;
    background: linear-gradient(120deg, #0b1d2d 0%, #0d2a3d 55%, #0c3a44 100%);
    border: 1px solid #1d3c52; border-radius: 18px;
    padding: 26px 30px 22px; margin-bottom: 18px;
  }}
  .hero::after {{
    content: ""; position: absolute; right: -80px; top: -120px;
    width: 360px; height: 360px; border-radius: 50%;
    background: radial-gradient(circle, rgba(46,196,182,.22), transparent 68%);
  }}
  .hero::before {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px;
    background: linear-gradient(180deg, {ACCENT}, {ACCENT_2});
  }}
  .kicker {{ font-size: 10.5px; letter-spacing: .22em; color: {ACCENT}; font-weight: 700; }}
  .hero h1 {{ margin: 7px 0 6px; font-size: 34px; line-height: 1.15; color: #f2fbfd; letter-spacing: .01em; }}
  .hero h1 span {{ color: {ACCENT}; }}
  .hero p {{ margin: 0; color: #a8c3cf; font-size: 13.5px; max-width: 760px; }}
  .badges {{ margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }}
  .badge {{
    font-size: 10.5px; font-weight: 700; letter-spacing: .06em;
    padding: 4px 10px; border-radius: 999px; border: 1px solid #24526a;
    color: #bfe6e2; background: rgba(46,196,182,.08);
  }}
  .badge.blue {{ color: #bcdcf5; background: rgba(90,169,230,.08); border-color: #244a68; }}
  .badge.amber {{ color: #f6d7a6; background: rgba(242,180,92,.08); border-color: #5d4524; }}

  .kpi-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin: 6px 0 16px; }}
  .kpi {{
    background: linear-gradient(180deg, #0d2030, #0b1a28);
    border: 1px solid {GRID}; border-radius: 13px; padding: 14px 16px 12px;
  }}
  .kpi .label {{ font-size: 10.5px; letter-spacing: .09em; color: {MUTED}; text-transform: uppercase; }}
  .kpi .value {{ font-size: 25px; font-weight: 800; color: #eef9fc; margin-top: 3px; line-height: 1.2; }}
  .kpi .sub {{ font-size: 10.5px; color: {MUTED}; margin-top: 2px; }}
  .kpi.good .value {{ color: {GREEN}; }}
  .kpi.warn .value {{ color: {AMBER}; }}
  .kpi.bad .value {{ color: {RED}; }}

  .section-title {{
    display: flex; align-items: baseline; gap: 10px;
    margin: 14px 0 8px; color: #eaf6fa; font-weight: 700; font-size: 15px;
  }}
  .section-title::before {{ content: ""; width: 4px; height: 15px; border-radius: 2px; background: {ACCENT}; }}
  .section-title small {{ color: {MUTED}; font-weight: 400; font-size: 11px; }}

  .panel {{
    background: #0b1a28; border: 1px solid {GRID}; border-radius: 13px;
    padding: 12px 14px 6px;
  }}
  .note {{
    background: rgba(46,196,182,.06); border-left: 3px solid {ACCENT};
    padding: 10px 13px; border-radius: 6px; color: #b9d7dc; font-size: 12px;
  }}
  .note.amber {{ background: rgba(242,180,92,.07); border-left-color: {AMBER}; color: #e7d2ac; }}

  div[data-testid="stMetric"] {{
    background: #0b1a28; border: 1px solid {GRID}; border-radius: 12px; padding: 10px 14px;
  }}
  div[data-testid="stMetricValue"] {{ color: #eef9fc; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
  .stTabs [data-baseweb="tab"] {{ background: #0b1a28; border-radius: 9px 9px 0 0; padding: 8px 16px; }}
  .stTabs [aria-selected="true"] {{ background: #123247; color: {ACCENT}; }}

  .stDataFrame {{ border: 1px solid {GRID}; border-radius: 10px; }}
  .stDownloadButton button, .stButton button {{
    background: #123247; color: #dff6f3; border: 1px solid #24526a; border-radius: 8px;
  }}
  .stDownloadButton button:hover, .stButton button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}

  @media (max-width: 1200px) {{
    .kpi-grid {{ grid-template-columns: repeat(3, 1fr); }}
  }}
  @media (max-width: 700px) {{
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .hero h1 {{ font-size: 26px; }}
  }}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def combine(datasets: list[StdfData]) -> StdfData:
    parts, results = [], []
    offset = 0
    bin_names: dict[int, str] = {}
    record_counts: Counter = Counter()
    errors: list[str] = []
    for d in datasets:
        p = d.parts.copy()
        r = d.results.copy()
        p["source"] = d.source
        r["source"] = d.source
        p["part_uid"] = p["part_uid"].astype(int) + offset
        r["part_uid"] = r["part_uid"].astype(int) + offset
        parts.append(p)
        results.append(r)
        offset += int(d.parts["part_uid"].max()) + 1 if not d.parts.empty else 1_000_000
        bin_names.update(d.bin_names)
        record_counts.update(d.record_counts)
        errors.extend(d.errors)

    parts_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    results_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    meta = dict(datasets[0].meta) if datasets else {}
    meta["files"] = [d.source for d in datasets]
    meta["record_counts"] = dict(record_counts)
    meta["errors"] = errors
    meta["multi_lot"] = True
    return StdfData(
        parts=parts_df,
        results=results_df,
        meta=meta,
        bin_names=bin_names,
        record_counts=record_counts,
        errors=errors,
        source=", ".join(d.source for d in datasets),
    )


@st.cache_data(show_spinner=False)
def load_demo() -> StdfData:
    paths = sorted(DEMO_DIR.glob("*.stdf"))
    if not paths:
        raise FileNotFoundError(f"no demo STDF in {DEMO_DIR}")
    return combine([parse_stdf(p) for p in paths])


@st.cache_data(show_spinner=False)
def load_upload(files: tuple[tuple[str, bytes], ...]) -> StdfData:
    return combine([parse_stdf_bytes(buf, name) for name, buf in files])


def style_fig(fig: go.Figure, height: int = 360, margin: dict | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL,
        font=dict(family="Inter, Microsoft YaHei, sans-serif", color=TEXT, size=11),
        margin=margin or dict(l=48, r=24, t=46, b=38),
        title=dict(font=dict(size=14, color="#eaf6fa")),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        hoverlabel=dict(bgcolor="#10293a", font_size=11, font_color=TEXT),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(size=10))
    return fig


def fmt_pct(v: float) -> str:
    return f"{v:.2f}%"


def kpi(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return (
        f"<div class='kpi {tone}'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div><div class='sub'>{sub}</div></div>"
    )


def section(title: str, note: str = "") -> None:
    st.markdown(
        f"<div class='section-title'>{title}"
        f"{' <small>' + note + '</small>' if note else ''}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# sidebar + data
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f"<div style='font-size:20px;font-weight:800;color:#eef9fc;letter-spacing:.02em'>"
        f"STDF <span style='color:{ACCENT}'>Yield Lab</span></div>"
        f"<div style='font-size:10.5px;color:{MUTED};letter-spacing:.12em;margin-bottom:14px'>"
        f"WAFER SORT ANALYTICS</div>",
        unsafe_allow_html=True,
    )
    uploads = st.file_uploader(
        "上传 STDF / STDF.GZ",
        type=["stdf", "std", "gz"],
        accept_multiple_files=True,
        help="支持未压缩 .stdf 与 gzip 压缩的 .stdf.gz；文件只在本地内存解析。",
    )
    st.caption("未上传时使用内置 3 个 lot 的演示数据。")

    if uploads:
        try:
            data = load_upload(tuple((f.name, f.getvalue()) for f in uploads))
            source_label = f"{len(uploads)} 个上传文件"
        except Exception as exc:
            st.error(f"解析失败：{exc}")
            st.stop()
    else:
        try:
            data = load_demo()
            source_label = "内置演示数据 · 3 lots"
        except Exception as exc:
            st.error(f"演示数据加载失败：{exc}")
            st.stop()

    st.divider()
    all_lots = sorted(data.parts["lot_id"].dropna().unique().tolist())
    all_wafers = sorted(data.parts["wafer_id"].dropna().unique().tolist())
    all_sites = sorted(data.parts["site"].dropna().unique().tolist()) if "site" in data.parts else []
    lots = st.multiselect("Lot", all_lots, default=all_lots)
    wafers = st.multiselect("Wafer", all_wafers, default=all_wafers)
    sites = st.multiselect("Site", all_sites, default=all_sites) if all_sites else []
    dedupe = st.toggle("去重复测（按 die 坐标）", value=False,
                       help="同一坐标多次测试时只保留最后一次，用于评估真实良率。")
    st.divider()
    st.caption(f"数据源：{source_label}")
    st.caption(f"文件：{data.source}")
    st.caption(f"STDF 记录：{sum(data.record_counts.values()):,}")

if not lots:
    st.warning("请至少选择一个 Lot。")
    st.stop()

parts = data.parts[data.parts["lot_id"].isin(lots)]
if wafers:
    parts = parts[parts["wafer_id"].isin(wafers)]
if sites and "site" in parts:
    parts = parts[parts["site"].isin(sites)]
parts = parts.copy()

if dedupe:
    parts_yield = unique_die(parts)
else:
    parts_yield = parts

results = data.results[data.results["part_uid"].isin(parts["part_uid"])].copy()
if dedupe:
    results = results[results["part_uid"].isin(parts_yield["part_uid"])]

if parts.empty:
    st.warning("当前筛选没有数据。")
    st.stop()

metrics = overview(StdfData(parts=parts_yield, results=results, meta=data.meta,
                            bin_names=data.bin_names, record_counts=data.record_counts))
stats = test_stats(results)
finite_cpk = stats["cpk"].replace([np.inf, -np.inf], np.nan) if not stats.empty else pd.Series(dtype=float)
worst_cpk = finite_cpk.min() if len(finite_cpk) and finite_cpk.notna().any() else float("nan")
worst_row = stats.loc[finite_cpk.idxmin()] if len(finite_cpk) and finite_cpk.notna().any() else None
yield_lot = yield_by_lot(parts_yield)
yield_wafer = yield_by_wafer(parts_yield)
bins = bin_pareto(parts_yield, data.bin_names)
fails = fail_pareto(results)

# ---------------------------------------------------------------------------
# header
# ---------------------------------------------------------------------------

lot_names = " / ".join(lots[:4]) + (" …" if len(lots) > 4 else "")
st.markdown(
    f"""
<div class="hero">
  <div class="kicker">SEMICONDUCTOR TEST ANALYTICS · STDF V4</div>
  <h1>STDF <span>Yield Lab</span></h1>
  <p>把 wafer sort 的原始 STDF 变成可行动的良率洞察：bin pareto、wafer map、测试项分布、Cpk 与跨 lot 趋势。</p>
  <div class="badges">
    <span class="badge">{len(lots)} LOT{'' if len(lots) == 1 else 'S'}</span>
    <span class="badge blue">{metrics['wafers']} WAFERS</span>
    <span class="badge blue">{metrics['parts']:,} DIE</span>
    <span class="badge amber">{len(stats)} TESTS</span>
    <span class="badge amber">{metrics['sites']} SITES</span>
    <span class="badge">{lot_names}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# navigation
# ---------------------------------------------------------------------------

page = st.radio(
    "导航",
    ["总览", "Wafer Map", "测试项", "趋势", "数据导出"],
    horizontal=True,
    label_visibility="collapsed",
)

# ---------------------------------------------------------------------------
# page: overview
# ---------------------------------------------------------------------------

if page == "总览":
    tone = "good" if metrics["yield"] >= 95 else ("warn" if metrics["yield"] >= 90 else "bad")
    cpk_tone = "good" if worst_cpk >= 1.33 else ("warn" if worst_cpk >= 1.0 else "bad")
    st.markdown(
        "<div class='kpi-grid'>"
        + kpi("总 Die 数", f"{metrics['parts']:,}", f"{metrics['wafers']} wafers")
        + kpi("良率", fmt_pct(metrics["yield"]), f"{metrics['passed']:,} pass / {metrics['failed']:,} fail", tone)
        + kpi("Lot / Wafer", f"{metrics['lots']} / {metrics['wafers']}", "多批次汇总")
        + kpi("测试项", f"{metrics['tests']}", f"{metrics['records']:,} 条 test record")
        + kpi("平均测试时间", f"{metrics['test_time']:.2f}s", "per die（含 index time 之外）")
        + kpi("最差 Cpk", f"{worst_cpk:.2f}" if np.isfinite(worst_cpk) else "∞",
              f"{worst_row['test_name']}" if worst_row is not None else "—", cpk_tone)
        + "</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1.35])
    with c1:
        section("各 Lot 良率", "目标线 95%")
        fig = go.Figure(
            go.Bar(
                x=yield_lot["lot_id"], y=yield_lot["yield"],
                marker=dict(
                    color=[GREEN if v >= 95 else (AMBER if v >= 90 else RED) for v in yield_lot["yield"]],
                    line=dict(width=0),
                ),
                text=[f"{v:.2f}%" for v in yield_lot["yield"]],
                textposition="outside",
                customdata=yield_lot[["parts", "passed"]],
                hovertemplate="%{x}<br>良率 %{y:.2f}%<br>%{customdata[1]} / %{customdata[0]} die<extra></extra>",
            )
        )
        fig.add_hline(y=95, line=dict(color=MUTED, width=1, dash="dash"))
        fig.update_layout(title="Yield by lot", showlegend=False, yaxis_range=[max(0, yield_lot["yield"].min() - 8), 101])
        st.plotly_chart(style_fig(fig, 330), width="stretch", config={"displayModeBar": False})

    with c2:
        section("Bin Pareto", "红色为 fail bin，折线为累计占比")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        colors = [GREEN if k == "PASS" else RED for k in bins["kind"]]
        fig.add_trace(
            go.Bar(x=bins["bin_name"], y=bins["count"], marker_color=colors,
                   text=bins["count"], textposition="outside",
                   hovertemplate="%{x}<br>%{y} die (%{customdata:.1f}%)<extra></extra>",
                   customdata=bins["pct"], name="die"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=bins["bin_name"], y=bins["cum_pct"], mode="lines+markers",
                       line=dict(color=ACCENT_2, width=2), marker=dict(size=6),
                       name="cum %", hovertemplate="%{x}<br>累计 %{y:.1f}%<extra></extra>"),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="die count", secondary_y=False)
        fig.update_yaxes(title_text="cumulative %", range=[0, 105], secondary_y=True)
        fig.update_layout(title="Bin distribution", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(style_fig(fig, 330), width="stretch", config={"displayModeBar": False})

    c3, c4 = st.columns([1.25, 1])
    with c3:
        section("Wafer 良率矩阵", "行 = lot，列 = wafer")
        pivot = yield_wafer.pivot(index="lot_id", columns="wafer_id", values="yield")
        counts = yield_wafer.pivot(index="lot_id", columns="wafer_id", values="parts")
        fig = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns, y=pivot.index,
                colorscale=[[0, "#5a1f28"], [0.5, "#6b4a1e"], [0.75, "#1d5a4a"], [1, "#1f8f68"]],
                zmin=max(0, np.nanmin(pivot.values) - 5), zmax=100,
                text=[[f"{pivot.loc[i, j]:.1f}%<br><span style='font-size:9px'>{int(counts.loc[i, j])} die</span>"
                       for j in pivot.columns] for i in pivot.index],
                texttemplate="%{text}", textfont=dict(size=11, color="#eef9fc"),
                hovertemplate="%{y} / %{x}<br>良率 %{z:.2f}%<extra></extra>",
                colorbar=dict(title="yield %", thickness=10),
            )
        )
        fig.update_layout(title="Wafer yield heatmap")
        st.plotly_chart(style_fig(fig, 300), width="stretch", config={"displayModeBar": False})

    with c4:
        section("失败测试项 Top", "按 fail record 数排序")
        if fails.empty:
            st.info("当前筛选下没有失败记录。")
        else:
            top = fails.head(8).iloc[::-1]
            fig = go.Figure(
                go.Bar(
                    x=top["fail_count"], y=top["test_name"], orientation="h",
                    marker_color=RED, text=top["fail_count"], textposition="outside",
                    hovertemplate="%{y}<br>%{x} fail records<extra></extra>",
                )
            )
            fig.update_layout(title="Fail pareto by test", showlegend=False)
            st.plotly_chart(style_fig(fig, 300), width="stretch", config={"displayModeBar": False})

    section("数据健康", "解析记录与告警")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("STDF 文件", len(data.meta.get("files", [data.source])))
    h1.caption(source_label)
    h2.metric("Records", f"{sum(data.record_counts.values()):,}")
    h2.caption("PTR / PRR / MIR …")
    h3.metric("解析告警", len(data.errors))
    h3.caption("truncated / malformed")
    h4.metric("数据模式", "DEMO" if not uploads else "UPLOAD")
    h4.caption("内置模拟数据" if not uploads else "用户上传")
    if data.errors:
        with st.expander("查看解析告警"):
            for err in data.errors[:30]:
                st.code(err)

# ---------------------------------------------------------------------------
# page: wafer map
# ---------------------------------------------------------------------------

elif page == "Wafer Map":
    section("Wafer Map", "选择 lot / wafer 查看 die 级分布")
    c1, c2, c3 = st.columns([1, 1, 2])
    lot_sel = c1.selectbox("Lot", sorted(parts["lot_id"].unique().tolist()))
    wafer_options = sorted(parts.loc[parts["lot_id"] == lot_sel, "wafer_id"].unique().tolist())
    wafer_sel = c2.selectbox("Wafer", wafer_options)
    color_by = c3.radio("着色", ["Pass / Fail", "Bin"], horizontal=True)

    wm = wafer_map_data(parts_yield, lot_sel, wafer_sel)
    if wm.empty:
        st.warning("该 wafer 没有数据。")
    else:
        bad = failing_tests_by_part(results[results["part_uid"].isin(wm["part_uid"])])
        wm = wm.merge(bad, on="part_uid", how="left")
        wm["failed_tests"] = wm["failed_tests"].fillna("—")
        wy = wm["passed"].mean() * 100
        n_fail = int((~wm["passed"]).sum())

        st.markdown(
            "<div class='kpi-grid'>"
            + kpi("Wafer", wafer_sel, lot_sel)
            + kpi("Die 数", f"{len(wm)}", f"site {', '.join(map(str, sorted(wm['site'].unique())))}" if "site" in wm else "")
            + kpi("良率", fmt_pct(wy), f"{int(wm['passed'].sum())} pass", "good" if wy >= 95 else ("warn" if wy >= 90 else "bad"))
            + kpi("Fail die", f"{n_fail}", "edge / scratch / random")
            + kpi("Bin 数", f"{wm['soft_bin'].nunique()}", "soft bin")
            + kpi("平均测试时间", f"{wm['test_time_s'].mean():.2f}s", "per die")
            + "</div>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([1.45, 1])
        with c1:
            fig = go.Figure()
            if color_by == "Pass / Fail":
                for passed, label, color in [(True, "PASS", GREEN), (False, "FAIL", RED)]:
                    sub = wm[wm["passed"] == passed]
                    if sub.empty:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=sub["x"], y=sub["y"], mode="markers", name=label,
                            marker=dict(symbol="square", size=12, color=color,
                                        line=dict(width=0.5, color="#07131e")),
                            customdata=np.stack([sub["part_id"], sub["soft_bin"], sub["failed_tests"]], axis=-1),
                            hovertemplate="%{customdata[0]}<br>x=%{x}, y=%{y}<br>bin %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
                        )
                    )
            else:
                palette = [ACCENT, RED, AMBER, ACCENT_2, "#b07cf0", "#ef8f5b", "#4fd1c5", "#9ad36a"]
                for i, b in enumerate(sorted(wm["soft_bin"].unique())):
                    sub = wm[wm["soft_bin"] == b]
                    label = data.bin_names.get(int(b), f"Bin {b}")
                    fig.add_trace(
                        go.Scatter(
                            x=sub["x"], y=sub["y"], mode="markers", name=f"{b} {label}",
                            marker=dict(symbol="square", size=12, color=palette[i % len(palette)],
                                        line=dict(width=0.5, color="#07131e")),
                            customdata=np.stack([sub["part_id"], sub["failed_tests"]], axis=-1),
                            hovertemplate="%{customdata[0]}<br>x=%{x}, y=%{y}<br>%{customdata[1]}<extra></extra>",
                        )
                    )
            theta = np.linspace(0, 2 * math.pi, 160)
            fig.add_trace(
                go.Scatter(
                    x=10.5 * np.cos(theta), y=10.5 * np.sin(theta), mode="lines",
                    line=dict(color="#3a5c72", width=1.2), showlegend=False, hoverinfo="skip",
                )
            )
            fig.update_layout(title=f"{lot_sel} / {wafer_sel} · {color_by}", showlegend=True,
                              xaxis_title="X", yaxis_title="Y")
            fig.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
            fig.update_xaxes(range=[-12, 12])
            st.plotly_chart(style_fig(fig, 560, margin=dict(l=40, r=20, t=46, b=40)),
                            width="stretch", config={"displayModeBar": False})

        with c2:
            section("Wafer Bin 分布")
            wb = wm.groupby(["soft_bin"]).size().reset_index(name="count").sort_values("count", ascending=False)
            wb["name"] = wb["soft_bin"].map(lambda b: data.bin_names.get(int(b), f"Bin {b}"))
            fig = go.Figure(
                go.Bar(
                    x=wb["count"], y=wb["name"], orientation="h",
                    marker_color=[GREEN if b == 1 else RED for b in wb["soft_bin"]],
                    text=wb["count"], textposition="outside",
                )
            )
            fig.update_layout(title="Bin count", showlegend=False)
            st.plotly_chart(style_fig(fig, 220, margin=dict(l=20, r=30, t=40, b=30)),
                            width="stretch", config={"displayModeBar": False})

            section("失败 Die 明细")
            bad_die = wm[~wm["passed"]][["part_id", "x", "y", "soft_bin", "failed_tests"]].head(40)
            st.dataframe(
                bad_die.rename(columns={"part_id": "Die", "soft_bin": "Bin", "failed_tests": "Failed tests"}),
                hide_index=True, width="stretch", height=300,
            )

        section("同 Lot 全部 Wafer", "一眼看空间良率分布")
        lot_wafers = sorted(parts_yield.loc[parts_yield["lot_id"] == lot_sel, "wafer_id"].unique().tolist())
        cols = st.columns(min(len(lot_wafers), 5))
        for i, w in enumerate(lot_wafers):
            sub = parts_yield[(parts_yield["lot_id"] == lot_sel) & (parts_yield["wafer_id"] == w)]
            fig = go.Figure()
            for passed, color in [(True, GREEN), (False, RED)]:
                ss = sub[sub["passed"] == passed]
                if ss.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=ss["x"], y=ss["y"], mode="markers", showlegend=False,
                    marker=dict(symbol="square", size=5, color=color),
                    hoverinfo="skip",
                ))
            fig.update_layout(title=f"{w} · {sub['passed'].mean() * 100:.1f}%",
                              margin=dict(l=4, r=4, t=28, b=4), height=190,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=PANEL,
                              font=dict(color=TEXT, size=9))
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1, autorange="reversed")
            cols[i % len(cols)].plotly_chart(fig, width="stretch",
                                             config={"displayModeBar": False}, key=f"mini_{w}")

# ---------------------------------------------------------------------------
# page: tests
# ---------------------------------------------------------------------------

elif page == "测试项":
    section("测试项分析", "分布 / limit / Cpk / 相关性")
    if stats.empty:
        st.info("没有测试数据。")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            test_name = st.selectbox("测试项", stats["test_name"].tolist())
            row = stats.loc[stats["test_name"] == test_name].iloc[0]
            vals = results.loc[results["test_name"] == test_name, "value"]
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=vals, nbinsx=36, marker_color=ACCENT, opacity=0.8,
                hovertemplate="value %{x}<br>count %{y}<extra></extra>", name="count",
            ))
            if pd.notna(row["lo"]):
                fig.add_vline(x=row["lo"], line=dict(color=RED, width=1.5, dash="dash"),
                              annotation_text=f"LSL {row['lo']:g}", annotation_position="top left")
            if pd.notna(row["hi"]):
                fig.add_vline(x=row["hi"], line=dict(color=RED, width=1.5, dash="dash"),
                              annotation_text=f"USL {row['hi']:g}", annotation_position="top right")
            fig.update_layout(title=f"{test_name} 分布 · n={len(vals):,}", showlegend=False,
                              xaxis_title=f"value ({row['units']})", yaxis_title="count")
            st.plotly_chart(style_fig(fig, 360), width="stretch", config={"displayModeBar": False})

        with c2:
            section("Cpk 排名", "红 <1.33，黄 1.33–1.67，绿 ≥1.67")
            s = stats.copy()
            s["plot"] = s["cpk"].replace([np.inf, -np.inf], np.nan).fillna(0)
            s = s.sort_values("plot")
            colors = [RED if v < 1.33 else (AMBER if v < 1.67 else GREEN) for v in s["plot"]]
            fig = go.Figure(go.Bar(
                x=s["plot"], y=s["test_name"], orientation="h", marker_color=colors,
                text=[("∞" if np.isinf(v) else f"{v:.2f}") for v in s["cpk"]],
                textposition="outside",
            ))
            fig.add_vline(x=1.33, line=dict(color=AMBER, dash="dash", width=1))
            fig.add_vline(x=1.67, line=dict(color=GREEN, dash="dot", width=1))
            fig.update_layout(title="Process capability", showlegend=False, xaxis_title="Cpk")
            st.plotly_chart(style_fig(fig, 360), width="stretch", config={"displayModeBar": False})

        section("测试项统计表")
        show = stats[["test_num", "test_name", "units", "n", "mean", "std", "min", "max",
                      "lo", "hi", "cpk", "fail", "fail_rate"]].copy()
        show.columns = ["#", "Test", "Unit", "N", "Mean", "σ", "Min", "Max",
                        "LSL", "USL", "Cpk", "Fail", "Fail %"]
        st.dataframe(
            show, hide_index=True, width="stretch", height=300,
            column_config={
                "Mean": st.column_config.NumberColumn(format="%.5g"),
                "σ": st.column_config.NumberColumn(format="%.4g"),
                "Min": st.column_config.NumberColumn(format="%.5g"),
                "Max": st.column_config.NumberColumn(format="%.5g"),
                "Cpk": st.column_config.NumberColumn(format="%.2f"),
                "Fail %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

        section("箱线图：按 Lot 比较", "看批间偏移与离散度")
        fig = go.Figure()
        for lot in sorted(results["lot_id"].unique()):
            sub = results[(results["test_name"] == test_name) & (results["lot_id"] == lot)]
            fig.add_trace(go.Box(y=sub["value"], name=lot, marker_color=ACCENT_2,
                                 boxmean=True, line=dict(width=1.2)))
        fig.update_layout(title=f"{test_name} by lot", showlegend=False, yaxis_title=row["units"])
        st.plotly_chart(style_fig(fig, 330), width="stretch", config={"displayModeBar": False})

        section("测试项相关性", "同 die 上两个参数的散点；用于发现设计/工艺关联")
        c1, c2 = st.columns(2)
        names = stats["test_name"].tolist()
        x_name = c1.selectbox("X 轴", names, index=1 if len(names) > 1 else 0)
        y_name = c2.selectbox("Y 轴", names, index=2 if len(names) > 2 else 0)
        pivot = test_matrix(results, [x_name, y_name]).dropna()
        if len(pivot) > 1:
            fig = go.Figure(go.Scatter(
                x=pivot[x_name], y=pivot[y_name], mode="markers",
                marker=dict(size=5, color=ACCENT, opacity=0.55),
                hovertemplate=f"{x_name} %{{x:.4g}}<br>{y_name} %{{y:.4g}}<extra></extra>",
            ))
            if len(pivot) >= 3:
                coef = np.polyfit(pivot[x_name], pivot[y_name], 1)
                xs = np.linspace(pivot[x_name].min(), pivot[x_name].max(), 50)
                fig.add_trace(go.Scatter(x=xs, y=np.polyval(coef, xs), mode="lines",
                                         line=dict(color=AMBER, width=2, dash="dash"),
                                         name=f"fit (r={pivot[x_name].corr(pivot[y_name]):.2f})"))
            fig.update_layout(title=f"{x_name} vs {y_name}", xaxis_title=x_name, yaxis_title=y_name)
            st.plotly_chart(style_fig(fig, 380), width="stretch", config={"displayModeBar": False})
        else:
            st.info("重叠 die 不足，无法画相关性。")

# ---------------------------------------------------------------------------
# page: trends
# ---------------------------------------------------------------------------

elif page == "趋势":
    section("跨 Lot / Wafer 趋势", "良率、测试项均值、fail rate 三个视角")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for lot in sorted(yield_wafer["lot_id"].unique()):
            sub = yield_wafer[yield_wafer["lot_id"] == lot]
            fig.add_trace(go.Scatter(
                x=sub["wafer_id"], y=sub["yield"], mode="lines+markers", name=lot,
                line=dict(width=2), marker=dict(size=7),
                hovertemplate=f"{lot} %{{x}}<br>yield %{{y:.2f}}%<extra></extra>",
            ))
        fig.add_hline(y=95, line=dict(color=MUTED, dash="dash", width=1))
        fig.update_layout(title="Wafer yield trend", yaxis_title="yield %", xaxis_title="wafer")
        st.plotly_chart(style_fig(fig, 340), width="stretch", config={"displayModeBar": False})

    with c2:
        fig = go.Figure()
        for lot in sorted(yield_lot["lot_id"].unique()):
            row = yield_lot[yield_lot["lot_id"] == lot].iloc[0]
            fig.add_trace(go.Bar(
                x=[lot], y=[row["yield"]], name=lot,
                marker_color=GREEN if row["yield"] >= 95 else AMBER,
                text=[f"{row['yield']:.2f}%"], textposition="outside",
            ))
        fig.add_hline(y=95, line=dict(color=MUTED, dash="dash", width=1))
        fig.update_layout(title="Lot yield", showlegend=False, yaxis_title="yield %",
                          yaxis_range=[max(0, yield_lot["yield"].min() - 5), 101])
        st.plotly_chart(style_fig(fig, 340), width="stretch", config={"displayModeBar": False})

    section("测试项均值热图", "按 lot 归一化，快速定位批间偏移")
    means = results.groupby(["test_name", "lot_id"])["value"].mean().unstack()
    norm = means.sub(means.mean(axis=1), axis=0).div(means.std(axis=1).replace(0, np.nan), axis=0)
    fig = go.Figure(go.Heatmap(
        z=norm.values, x=norm.columns, y=norm.index,
        colorscale="RdBu", zmid=0, zmin=-2.5, zmax=2.5,
        colorbar=dict(title="z-score", thickness=10),
        hovertemplate="%{y}<br>%{x}<br>z = %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(title="Test mean shift by lot (z-score)")
    st.plotly_chart(style_fig(fig, max(300, len(norm) * 30 + 80)),
                    width="stretch", config={"displayModeBar": False})

    section("Fail rate 热图", "每个测试项在每个 lot 的失败比例")
    fail_rate = results.groupby(["test_name", "lot_id"])["failed"].mean().unstack() * 100
    fig = go.Figure(go.Heatmap(
        z=fail_rate.values, x=fail_rate.columns, y=fail_rate.index,
        colorscale=[[0, "#0c1c2b"], [0.4, "#5a3a1f"], [1, "#b33a3a"]],
        text=[[f"{v:.2f}%" for v in row] for row in fail_rate.values],
        texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title="fail %", thickness=10),
        hovertemplate="%{y}<br>%{x}<br>%{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(title="Fail rate by test × lot")
    st.plotly_chart(style_fig(fig, max(300, len(fail_rate) * 30 + 80)),
                    width="stretch", config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# page: data
# ---------------------------------------------------------------------------

else:
    section("数据导出", "筛选后的 die、test record 与解析元数据")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Die rows", f"{len(parts_yield):,}")
    c2.metric("Test records", f"{len(results):,}")
    c3.metric("Lots", len(lots))
    c4.metric("Wafers", metrics["wafers"])

    st.markdown("<div class='note'>导出的 CSV 采用 UTF-8-SIG，Excel 可直接打开。Die 表包含 bin、坐标、失败项；Test 表为长表，适合 pandas 二次分析。</div>",
                unsafe_allow_html=True)

    bad = failing_tests_by_part(results)
    die_df = parts_yield.merge(bad, on="part_uid", how="left")
    die_df["failed_tests"] = die_df["failed_tests"].fillna("")
    die_df["status"] = np.where(die_df["passed"], "PASS", "FAIL")
    show_cols = ["lot_id", "wafer_id", "part_id", "x", "y", "site", "hard_bin",
                 "soft_bin", "bin_name", "status", "test_time_s", "failed_tests"]
    show_cols = [c for c in show_cols if c in die_df.columns]
    st.dataframe(die_df[show_cols].head(500), hide_index=True, width="stretch", height=360)

    c1, c2 = st.columns(2)
    c1.download_button(
        "下载 Die CSV",
        die_df[show_cols].to_csv(index=False).encode("utf-8-sig"),
        file_name="stdf_die_summary.csv",
        mime="text/csv",
        width="stretch",
    )
    c2.download_button(
        "下载 Test Record CSV",
        results.to_csv(index=False).encode("utf-8-sig"),
        file_name="stdf_test_records.csv",
        mime="text/csv",
        width="stretch",
    )

    section("STDF 元数据")
    meta_rows = [
        ("文件", data.source),
        ("Part type", data.meta.get("part_type", "—")),
        ("Tester", data.meta.get("tester", "—")),
        ("Job", data.meta.get("job", "—")),
        ("Test temp", data.meta.get("test_temp", "—")),
        ("Start time", data.meta.get("start_time", "—")),
        ("Endian", data.meta.get("endian", "—")),
        ("File size", f"{data.meta.get('file_size_mb', '—')} MB"),
        ("Records", f"{sum(data.record_counts.values()):,}"),
    ]
    st.dataframe(pd.DataFrame(meta_rows, columns=["Key", "Value"]),
                 hide_index=True, width="stretch")

    with st.expander("支持的 STDF 记录与已知限制"):
        st.markdown(
            """
- 已解析：FAR、ATR、MIR、MRR、PCR、HBR、SDR、PIR、PRR、PTR、MPR、TSR、WIR、WRR、WCR、TIR、GDR、DTR。
- 仅支持 STDF V4；小端为主，若首记录长度异常会自动尝试大端。
- `MPR` 多结果记录目前只按单结果读取；`DTR` 仅保留原文，不做 flat-data 展开。
- 对 `LO_LIMIT=0` / `HI_LIMIT=0` 采用"该侧无 limit"的常见约定；若你的测试程序确实使用 0 作为 limit，需要按业务调整。
- 复测去重按 `lot_id + wafer_id + x + y` 保留最后一次；没有坐标的数据集请关闭去重。
            """
        )

"""STDF V4 parser (little-endian, with big-endian fallback).

Supports the records needed for yield / wafer-map / test-item analysis:
FAR, ATR, MIR, MRR, PCR, HBR, SDR, PIR, PRR, PTR, MPR, TSR, WIR, WRR, WCR,
TIR, GDR, DTR. Unknown records are skipped safely.

Only the standard library is required for parsing; pandas is used for the
tabular output.
"""

from __future__ import annotations

import gzip
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# name -> struct format
NUM = {
    "U1": "B",
    "U2": "H",
    "U4": "I",
    "I1": "b",
    "I2": "h",
    "I4": "i",
    "R4": "f",
    "B1": "B",
}

# Record type/sub -> field specification.
# C = variable-length string (1-byte length), C1 = fixed one character,
# Bn = remaining bytes, U1n = array of U1 with count from SITE_CNT.
SPECS: dict[tuple[int, int], list[tuple[str, str]]] = {
    (0, 10): [("CPU_TYPE", "U1"), ("STDF_VER", "U1")],
    (0, 20): [("MOD_TIM", "U4"), ("CMD_LINE", "C")],
    (1, 10): [
        ("SETUP_T", "U4"), ("START_T", "U4"), ("STAT_NUM", "U1"),
        ("MODE_COD", "C1"), ("RTST_COD", "C1"), ("PROT_COD", "C1"),
        ("BURN_TIM", "U2"), ("CMOD_COD", "U1"), ("LOT_ID", "C"),
        ("PART_TYP", "C"), ("NODE_NAM", "C"), ("TSTR_TYP", "C"),
        ("JOB_NAM", "C"), ("JOB_REV", "C"), ("SBLOT_ID", "C"),
        ("OPER_NAM", "C"), ("EXEC_TYP", "C"), ("EXEC_VER", "C"),
        ("TEST_COD", "C"), ("TST_TEMP", "C"), ("USER_TXT", "C"),
        ("AUX_FILE", "C"), ("PKG_TYP", "C"), ("FAMLY_ID", "C"),
        ("DATE_COD", "C"), ("FACIL_ID", "C"), ("FLOOR_ID", "C"),
        ("PROC_ID", "C"), ("OPER_FRQ", "C"), ("SPEC_NAM", "C"),
        ("SPEC_VER", "C"), ("FLOW_ID", "C"), ("SETUP_ID", "C"),
        ("DSGN_REV", "C"), ("ENG_ID", "C"), ("ROM_COD", "C"),
        ("SERL_NUM", "C"), ("SUPR_NAM", "C"),
    ],
    (1, 20): [("FINISH_T", "U4"), ("DISP_COD", "C1"), ("USR_DESC", "C"), ("EXC_DESC", "C")],
    (1, 30): [
        ("HEAD_NUM", "U1"), ("SITE_NUM", "U1"), ("PART_CNT", "U4"),
        ("RTST_CNT", "U4"), ("ABRT_CNT", "U4"), ("GOOD_CNT", "U4"), ("FUNC_CNT", "U4"),
    ],
    (1, 40): [
        ("HEAD_NUM", "U1"), ("SITE_NUM", "U1"), ("HB_NUM", "U2"), ("HB_CNT", "U4"),
        ("XY_MAN", "C1"), ("HB_MODE", "C1"), ("HB_POS_X", "I2"), ("HB_POS_Y", "I2"),
        ("HB_SRC", "C1"), ("HB_SOFT", "C"),
    ],
    (1, 80): [
        ("HEAD_NUM", "U1"), ("SITE_GRP", "U1"), ("SITE_CNT", "U1"), ("SITE_NUM", "U1n"),
        ("HAND_TYP", "C1"), ("CARD_TYP", "C1"), ("CARD_ID", "C"), ("LOAD_TYP", "C1"),
        ("LOAD_ID", "C"), ("DIB_TYP", "C1"), ("DIB_ID", "C"), ("CABL_TYP", "C1"),
        ("CABL_ID", "C"), ("CONT_LEN", "U1"), ("CONT_TYP", "C1"), ("CONT_ID", "C"),
        ("LASR_TYP", "C1"), ("LASR_ID", "C"), ("EXTR_TYP", "C1"), ("EXTR_ID", "C"),
    ],
    (2, 10): [("HEAD_NUM", "U1"), ("START_T", "U4")],
    (2, 20): [
        ("HEAD_NUM", "U1"), ("FINISH_T", "U4"), ("PART_CNT", "U4"),
        ("RTST_CNT", "U4"), ("ABRT_CNT", "U4"), ("GOOD_CNT", "U4"), ("FUNC_CNT", "U4"),
    ],
    (2, 30): [("WAFER_ID", "C")],
    (5, 10): [("HEAD_NUM", "U1"), ("SITE_NUM", "U1")],
    (5, 20): [
        ("HEAD_NUM", "U1"), ("SITE_NUM", "U1"), ("PART_FLG", "B1"), ("NUM_TEST", "U2"),
        ("HARD_BIN", "U2"), ("SOFT_BIN", "U2"), ("X_COORD", "I2"), ("Y_COORD", "I2"),
        ("TEST_T", "U4"), ("PART_ID", "C"), ("PART_TXT", "C"), ("PART_FIX", "Bn"),
    ],
    (5, 30): [
        ("TEST_NUM", "U4"), ("HEAD_NUM", "U1"), ("SITE_NUM", "U1"),
        ("TEST_FLG", "B1"), ("PARM_FLG", "B1"), ("RESULT", "R4"), ("TEST_TXT", "C"),
        ("ALARM_ID", "C"), ("OPT_FLAG", "B1"), ("RES_SCAL", "I1"),
        ("LLM_SCAL", "I1"), ("HLM_SCAL", "I1"), ("LO_LIMIT", "R4"),
        ("HI_LIMIT", "R4"), ("UNITS", "C"), ("C_RESFMT", "C"),
        ("C_LLMFMT", "C"), ("C_HLMFMT", "C"), ("LO_SPEC", "R4"), ("HI_SPEC", "R4"),
    ],
    (5, 50): [
        ("TEST_NUM", "U4"), ("HEAD_NUM", "U1"), ("SITE_NUM", "U1"),
        ("TEST_FLG", "B1"), ("PARM_FLG", "B1"), ("RTN_RSLT", "R4"), ("TEST_TXT", "C"),
        ("ALARM_ID", "C"), ("OPT_FLAG", "B1"), ("RES_SCAL", "I1"),
        ("LLM_SCAL", "I1"), ("HLM_SCAL", "I1"), ("LO_LIMIT", "R4"),
        ("HI_LIMIT", "R4"), ("UNITS", "C"), ("C_RESFMT", "C"),
        ("C_LLMFMT", "C"), ("C_HLMFMT", "C"), ("LO_SPEC", "R4"), ("HI_SPEC", "R4"),
    ],
    (10, 10): [("HEAD_NUM", "U1"), ("SITE_NUM", "U1")],
    (10, 30): [
        ("HEAD_NUM", "U1"), ("SITE_NUM", "U1"), ("TEST_NUM", "U4"),
        ("EXEC_CNT", "U4"), ("FAIL_CNT", "U4"), ("ALRM_CNT", "U4"),
    ],
    (50, 10): [("GEN_DATA", "Bn")],
    (50, 30): [("TEXT_DAT", "Bn")],
}

RECORD_NAMES = {
    (0, 10): "FAR", (0, 20): "ATR", (1, 10): "MIR", (1, 20): "MRR",
    (1, 30): "PCR", (1, 40): "HBR", (1, 80): "SDR", (2, 10): "WIR",
    (2, 20): "WRR", (2, 30): "WCR", (5, 10): "PIR", (5, 20): "PRR",
    (5, 30): "PTR", (5, 50): "MPR", (10, 10): "TIR", (10, 30): "TSR",
    (50, 10): "GDR", (50, 30): "DTR",
}


@dataclass
class StdfData:
    parts: pd.DataFrame
    results: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)
    bin_names: dict[int, str] = field(default_factory=dict)
    record_counts: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)
    source: str = ""

    @property
    def lots(self) -> list[str]:
        return sorted(self.parts["lot_id"].dropna().unique().tolist())

    @property
    def wafer_count(self) -> int:
        return int(self.parts[["lot_id", "wafer_id"]].drop_duplicates().shape[0])

    @property
    def yield_pct(self) -> float:
        if self.parts.empty:
            return 0.0
        return float(self.parts["passed"].mean() * 100)


def _read_bytes(path: Path) -> bytes:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rb") as fh:
            return fh.read()
    return path.read_bytes()


def _decode_text(raw: bytes) -> str:
    return raw.decode("latin-1", errors="replace").strip("\x00 ")


def _parse_payload(payload: bytes, spec: list[tuple[str, str]], endian: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    off = 0
    for name, kind in spec:
        if kind == "C":
            if off >= len(payload):
                break
            n = payload[off]
            off += 1
            values[name] = _decode_text(payload[off:off + n])
            off += n
        elif kind == "C1":
            values[name] = _decode_text(payload[off:off + 1])
            off += 1
        elif kind == "Bn":
            values[name] = payload[off:]
            off = len(payload)
        elif kind == "U1n":
            n = int(values.get("SITE_CNT", 0))
            values[name] = list(payload[off:off + n])
            off += n
        else:
            fmt = NUM[kind]
            size = struct.calcsize(fmt)
            if off + size > len(payload):
                break
            values[name] = struct.unpack_from(f"{endian}{fmt}", payload, off)[0]
            off += size
    return values


def _detect_endian(buf: bytes) -> str:
    if len(buf) < 6:
        return "<"
    for endian in ("<", ">"):
        length, typ, sub = struct.unpack_from(f"{endian}HBB", buf, 0)
        if typ == 0 and sub == 10 and 2 <= length <= len(buf) - 4 and length <= 1024:
            return endian
    return "<"


def _as_time(seconds: int | None) -> str:
    if not seconds:
        return ""
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (OverflowError, OSError, ValueError):
        return ""


def parse_stdf(
    path: str | Path | None = None,
    *,
    data: bytes | None = None,
    source_name: str = "",
) -> StdfData:
    """Parse a STDF path, or raw bytes when `data` is given."""
    if data is None:
        if path is None:
            raise ValueError("path or data is required")
        path = Path(path)
        buf = _read_bytes(path)
        source_name = source_name or path.name
    else:
        buf = data
    endian = _detect_endian(buf)
    records: Counter = Counter()
    errors: list[str] = []
    meta: dict[str, Any] = {}
    bin_names: dict[int, str] = {}

    lot_id = ""
    wafer_id = ""
    current: dict[tuple[int, int], dict[str, Any]] = {}
    pending: dict[tuple[int, int], dict[str, Any]] = defaultdict(dict)
    parts: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    part_fail: dict[int, bool] = defaultdict(bool)
    part_seq = 0
    pos = 0
    n = len(buf)

    while pos + 4 <= n:
        rec_len, rec_typ, rec_sub = struct.unpack_from(f"{endian}HBB", buf, pos)
        if pos + 4 + rec_len > n:
            errors.append(f"truncated record at offset {pos}: len={rec_len}")
            break
        payload = buf[pos + 4:pos + 4 + rec_len]
        key = (rec_typ, rec_sub)
        records[key] += 1
        values = _parse_payload(payload, SPECS.get(key, []), endian) if key in SPECS else {}
        pos += 4 + rec_len

        if key == (1, 10):  # MIR
            lot_id = str(values.get("LOT_ID", "") or lot_id)
            meta.update(
                {
                    "lot_id": lot_id,
                    "part_type": values.get("PART_TYP", ""),
                    "tester": values.get("TSTR_TYP", ""),
                    "node": values.get("NODE_NAM", ""),
                    "job": values.get("JOB_NAM", ""),
                    "test_temp": values.get("TST_TEMP", ""),
                    "start_time": _as_time(values.get("START_T")),
                    "endian": "little" if endian == "<" else "big",
                }
            )
        elif key == (2, 10):  # WIR
            wafer_id = ""
        elif key == (2, 30):  # WCR
            wafer_id = str(values.get("WAFER_ID", "") or "")
        elif key == (1, 40):  # HBR
            hb = int(values.get("HB_NUM", 0))
            label = str(values.get("HB_SOFT", "") or "").strip()
            bin_names[hb] = label or f"Bin {hb}"
        elif key == (5, 10):  # PIR
            head = int(values.get("HEAD_NUM", 1))
            site = int(values.get("SITE_NUM", 1))
            pending[(head, site)] = {
                "part_uid": part_seq,
                "lot_id": lot_id,
                "wafer_id": wafer_id,
                "head": head,
                "site": site,
                "part_start": values.get("START_T", 0),
            }
            current[(head, site)] = pending[(head, site)]
            part_seq += 1
        elif key == (5, 30):  # PTR
            head = int(values.get("HEAD_NUM", 1))
            site = int(values.get("SITE_NUM", 1))
            part = current.get((head, site))
            if part is None:
                part = {"part_uid": part_seq, "lot_id": lot_id, "wafer_id": wafer_id,
                        "head": head, "site": site}
                current[(head, site)] = part
                part_seq += 1
            test_flg = int(values.get("TEST_FLG", 0))
            if test_flg & 0x02 or test_flg & 0x10:  # invalid / not executed
                continue
            res_scal = int(values.get("RES_SCAL", 0) or 0)
            llm_scal = int(values.get("LLM_SCAL", 0) or 0)
            hlm_scal = int(values.get("HLM_SCAL", 0) or 0)
            scale = 10.0 ** res_scal
            lo = values.get("LO_LIMIT")
            hi = values.get("HI_LIMIT")
            result = float(values.get("RESULT", float("nan")))
            if lo is not None:
                lo = float(lo) * (10.0 ** llm_scal)
            if hi is not None:
                hi = float(hi) * (10.0 ** hlm_scal)
            # STDF convention: a zero limit on one side often means "no limit".
            if lo == 0.0 and hi is not None and hi > 0:
                lo = None
            if hi == 0.0 and lo is not None and lo != 0:
                hi = None
            if lo == 0.0 and hi == 0.0:
                lo = hi = None
            results.append(
                {
                    "lot_id": part.get("lot_id", lot_id),
                    "wafer_id": part.get("wafer_id", wafer_id),
                    "part_uid": part["part_uid"],
                    "head": head,
                    "site": site,
                    "test_num": int(values.get("TEST_NUM", 0)),
                    "test_name": str(values.get("TEST_TXT", "") or f"Test {values.get('TEST_NUM', 0)}"),
                    "value": result * scale,
                    "lo_limit": lo,
                    "hi_limit": hi,
                    "units": str(values.get("UNITS", "") or ""),
                    "test_flg": test_flg,
                    "failed": bool(test_flg & 0x80),
                }
            )
            if test_flg & 0x80:
                part_fail[part["part_uid"]] = True
        elif key == (5, 20):  # PRR
            head = int(values.get("HEAD_NUM", 1))
            site = int(values.get("SITE_NUM", 1))
            part = current.pop((head, site), None) or pending.get((head, site), {})
            part_flg = int(values.get("PART_FLG", 0))
            hard_bin = int(values.get("HARD_BIN", 0))
            soft_bin = int(values.get("SOFT_BIN", 0))
            if part_flg & 0x10:
                passed = hard_bin in (1, 0)
            else:
                passed = not bool(part_flg & 0x08)
            if part_fail.get(part.get("part_uid", -1)):
                passed = False
            parts.append(
                {
                    "lot_id": part.get("lot_id", lot_id),
                    "wafer_id": part.get("wafer_id", wafer_id),
                    "part_uid": part.get("part_uid", part_seq),
                    "head": head,
                    "site": site,
                    "x": int(values.get("X_COORD", 0)),
                    "y": int(values.get("Y_COORD", 0)),
                    "hard_bin": hard_bin,
                    "soft_bin": soft_bin,
                    "part_flg": part_flg,
                    "passed": passed,
                    "num_test": int(values.get("NUM_TEST", 0)),
                    "test_time_s": float(values.get("TEST_T", 0) or 0) / 1000.0,
                    "part_id": str(values.get("PART_ID", "") or ""),
                }
            )
            pending.pop((head, site), None)
        elif key == (1, 30):  # PCR
            meta["part_count_summary"] = {
                "parts": values.get("PART_CNT", 0),
                "good": values.get("GOOD_CNT", 0),
                "retest": values.get("RTST_CNT", 0),
                "aborted": values.get("ABRT_CNT", 0),
            }

    parts_df = pd.DataFrame(parts)
    results_df = pd.DataFrame(results)
    if parts_df.empty:
        parts_df = pd.DataFrame(
            columns=["lot_id", "wafer_id", "part_uid", "head", "site", "x", "y",
                     "hard_bin", "soft_bin", "part_flg", "passed", "num_test",
                     "test_time_s", "part_id"]
        )
    else:
        parts_df["passed"] = parts_df["passed"].astype(bool)
        # A part is failed if any PTR failed, even if PART_FLG was inconsistent.
        failed_uids = set(results_df.loc[results_df["failed"], "part_uid"]) if not results_df.empty else set()
        if failed_uids:
            parts_df.loc[parts_df["part_uid"].isin(failed_uids), "passed"] = False
        parts_df["bin_name"] = parts_df["soft_bin"].map(
            lambda b: bin_names.get(int(b), f"Bin {b}")
        )

    if results_df.empty:
        results_df = pd.DataFrame(
            columns=["lot_id", "wafer_id", "part_uid", "head", "site", "test_num",
                     "test_name", "value", "lo_limit", "hi_limit", "units",
                     "test_flg", "failed"]
        )
    else:
        results_df["failed"] = results_df["failed"].astype(bool)
        results_df["value"] = pd.to_numeric(results_df["value"], errors="coerce")
        results_df = results_df.dropna(subset=["value"])

    meta.setdefault("lot_id", lot_id)
    meta["record_counts"] = dict(records)
    meta["file_size_mb"] = round(len(buf) / 1024 / 1024, 2)
    meta["source"] = source_name or "uploaded.stdf"
    meta["n_records"] = sum(records.values())

    return StdfData(
        parts=parts_df,
        results=results_df,
        meta=meta,
        bin_names=bin_names,
        record_counts=records,
        errors=errors,
        source=source_name or "uploaded.stdf",
    )


def parse_stdf_bytes(buf: bytes, source_name: str = "uploaded.stdf") -> StdfData:
    """Parse an in-memory STDF file (e.g. a Streamlit upload)."""
    if buf[:2] == bytes([0x1F, 0x8B]):
        buf = gzip.decompress(buf)
    return parse_stdf(data=buf, source_name=source_name)

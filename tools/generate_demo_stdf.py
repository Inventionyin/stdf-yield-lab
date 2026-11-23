#!/usr/bin/env python3
"""Generate realistic demo STDF V4 files for the dashboard.

The data is synthetic but shaped like a wafer-sort dataset:
3 lots x 5 wafers x ~300 die, 10 tests, 4 sites, edge/scratch fail patterns.

Usage:
    python tools/generate_demo_stdf.py --out data
"""

from __future__ import annotations

import argparse
import math
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# STDF primitives
# ---------------------------------------------------------------------------

def rec(typ: int, sub: int, payload: bytes) -> bytes:
    return struct.pack("<HBB", len(payload), typ, sub) + payload


def u1(v: int) -> bytes:
    return struct.pack("<B", int(v))


def u2(v: int) -> bytes:
    return struct.pack("<H", int(v))


def u4(v: int) -> bytes:
    return struct.pack("<I", int(v))


def i2(v: int) -> bytes:
    return struct.pack("<h", int(v))


def r4(v: float) -> bytes:
    return struct.pack("<f", float(v))


def c(text: str) -> bytes:
    raw = str(text).encode("latin-1", errors="replace")[:255]
    return bytes([len(raw)]) + raw


def c1(text: str) -> bytes:
    raw = str(text).encode("latin-1", errors="replace")[:1]
    return raw.ljust(1, b" ")


def ptr(test_num: int, head: int, site: int, value: float, test_txt: str,
        lo: float | None, hi: float | None, units: str, failed: bool) -> bytes:
    flags = 0x80 if failed else 0
    payload = b"".join([
        u4(test_num), u1(head), u1(site), u1(flags), u1(0), r4(value),
        c(test_txt), c(""), u1(0), struct.pack("<b", 0), struct.pack("<b", 0),
        struct.pack("<b", 0), r4(lo if lo is not None else 0.0),
        r4(hi if hi is not None else 0.0), c(units), c("%.6g"), c("%.6g"),
        c("%.6g"), r4(lo if lo is not None else 0.0), r4(hi if hi is not None else 0.0),
    ])
    return rec(5, 30, payload)


# ---------------------------------------------------------------------------
# test definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestDef:
    num: int
    name: str
    units: str
    lo: float | None
    hi: float | None
    mean: float
    sigma: float
    bin: int


TESTS = [
    TestDef(1, "Continuity", "V", 0.3, 0.9, 0.65, 0.012, 6),
    TestDef(2, "VOUT_10mA", "V", 3.270, 3.330, 3.300, 0.0065, 2),
    TestDef(3, "VOUT_100mA", "V", 3.270, 3.330, 3.299, 0.0068, 2),
    TestDef(4, "VOUT_500mA", "V", 3.270, 3.330, 3.296, 0.0075, 2),
    TestDef(5, "VOUT_1A", "V", 3.265, 3.335, 3.293, 0.0085, 2),
    TestDef(6, "LineReg_mV", "mV", 0.0, 6.0, 1.8, 0.70, 3),
    TestDef(7, "LoadReg_mV", "mV", 0.0, 12.0, 4.5, 1.50, 3),
    TestDef(8, "Dropout_V", "V", 0.0, 1.30, 1.02, 0.050, 4),
    TestDef(9, "IQ_mA", "mA", 0.0, 10.0, 5.4, 0.50, 5),
    TestDef(10, "PSRR_dB", "dB", 62.0, None, 70.0, 1.80, 7),
]

BIN_NAMES = {
    1: "PASS",
    2: "VOUT_FAIL",
    3: "REG_FAIL",
    4: "DROPOUT_FAIL",
    5: "IQ_FAIL",
    6: "CONT_FAIL",
    7: "PSRR_FAIL",
}
FAIL_WEIGHTS = {2: 0.25, 3: 0.20, 4: 0.15, 5: 0.15, 6: 0.10, 7: 0.15}


# ---------------------------------------------------------------------------
# wafer / lot simulation
# ---------------------------------------------------------------------------

@dataclass
class Die:
    x: int
    y: int
    site: int
    values: dict[int, float]
    fail_bin: int
    test_time: float


@dataclass
class Wafer:
    wafer_id: str
    die: list[Die]


@dataclass
class Lot:
    lot_id: str
    start: datetime
    wafers: list[Wafer]


def wafer_coordinates(radius: float = 10.5) -> list[tuple[int, int]]:
    coords = []
    for y in range(-11, 12):
        for x in range(-11, 12):
            if x * x + y * y <= radius * radius:
                if y <= -9 and abs(x) <= 2:  # notch
                    continue
                coords.append((x, y))
    return coords


def make_lot(rng: np.random.Generator, lot_index: int, lot_id: str,
             n_wafers: int, wafer_offset: float, start: datetime) -> Lot:
    wafers: list[Wafer] = []
    coords = wafer_coordinates()

    for w in range(n_wafers):
        wafer_id = f"W{w + 1:02d}"
        wafer_bias = rng.normal(0, 0.0032)
        wafer_sigma = rng.uniform(0.85, 1.20)
        scratch = (lot_index == 1 and w == 2)  # one dramatic scratch wafer

        dies: list[Die] = []
        for x, y in coords:
            r = math.hypot(x, y) / 10.5
            theta = math.atan2(y, x)
            site = ((x + y) % 4) + 1

            # spatial signature: radial roll-off + gentle wave
            spatial = -0.003 * (r ** 2 - 0.45) + 0.0012 * math.cos(3 * theta)
            quality = rng.normal(0, 1)

            values: dict[int, float] = {}
            for t in TESTS:
                sigma = t.sigma * wafer_sigma
                val = t.mean + (spatial if t.num in (2, 3, 4, 5) else 0.0)
                if t.num in (2, 3, 4, 5):
                    val += wafer_bias
                val += rng.normal(0, sigma)
                values[t.num] = val

            # edge + scratch fail probability
            p_fail = 0.008 + 0.055 * (r ** 4)
            if scratch and abs(y - 2) <= 1 and x > -4:
                p_fail = 0.85
            fail_bin = 1
            if rng.random() < p_fail:
                fail_bin = rng.choice(
                    list(FAIL_WEIGHTS), p=list(FAIL_WEIGHTS.values())
                )
                t = next(t for t in TESTS if t.bin == fail_bin)
                if t.hi is not None:
                    values[t.num] = t.hi + abs(rng.normal(0, max(t.sigma, 0.01))) + 0.002
                else:
                    values[t.num] = (t.lo or 0) - abs(rng.normal(0, max(t.sigma, 0.01))) - 0.002

            # Make the bin consistent with the actual first failing test.
            actual_bin = 1
            for t in TESTS:
                v = values[t.num]
                if (t.lo is not None and v < t.lo) or (t.hi is not None and v > t.hi):
                    actual_bin = t.bin
                    break
            if actual_bin != 1:
                fail_bin = actual_bin

            dies.append(
                Die(
                    x=x, y=y, site=site, values=values, fail_bin=int(fail_bin),
                    test_time=float(np.clip(rng.normal(0.86, 0.08), 0.55, 1.25)),
                )
            )
        wafers.append(Wafer(wafer_id, dies))
    return Lot(lot_id, start, wafers)


# ---------------------------------------------------------------------------
# STDF writer
# ---------------------------------------------------------------------------

def write_lot(fh, lot: Lot, lot_index: int) -> None:
    start_ts = int(lot.start.replace(tzinfo=timezone.utc).timestamp())
    mir = b"".join([
        u4(start_ts), u4(start_ts), u1(1), c1("P"), c1(" "), c1(" "),
        u2(0), u1(0),
        c(lot.lot_id), c("AMS1117-3.3"), c("SIM-WAFER-SORT"), c("SIMAT"),
        c("AMS1117_FT"), c("1.0"), c(lot.lot_id), c("SIM"),
        c("STDF"), c("4.0"), c("FT"), c("25"),
    ])
    fh.write(rec(1, 10, mir))

    # bin definitions with counts
    all_die = [d for w in lot.wafers for d in w.die]
    for b in sorted(BIN_NAMES):
        count = sum(1 for d in all_die if d.fail_bin == b)
        hbr = b"".join([
            u1(1), u1(1), u2(b), u4(count), c1(" "), c1("F"),
            i2(0), i2(0), c1(" "), c(BIN_NAMES[b]),
        ])
        fh.write(rec(1, 40, hbr))

    good_total = 0
    part_total = 0
    for w in lot.wafers:
        fh.write(rec(2, 10, u1(1) + u4(start_ts)))
        fh.write(rec(2, 30, c(w.wafer_id)))
        good = 0
        for p, d in enumerate(w.die):
            test_time_ms = int(d.test_time * 1000)
            fh.write(rec(5, 10, u1(1) + u1(d.site)))
            for t in TESTS:
                val = d.values[t.num]
                failed = False
                if t.lo is not None and val < t.lo:
                    failed = True
                if t.hi is not None and val > t.hi:
                    failed = True
                fh.write(ptr(t.num, 1, d.site, val, t.name, t.lo, t.hi, t.units, failed))
            passed = d.fail_bin == 1
            if passed:
                good += 1
            part_flg = 0 if passed else 0x08
            prr = b"".join([
                u1(1), u1(d.site), u1(part_flg), u2(len(TESTS)),
                u2(d.fail_bin), u2(d.fail_bin), i2(d.x), i2(d.y),
                u4(test_time_ms), c(f"{lot.lot_id}-{w.wafer_id}-{p:04d}"),
                c(""), b"",
            ])
            fh.write(rec(5, 20, prr))
        wrr = b"".join([
            u1(1), u4(start_ts), u4(len(w.die)), u4(0), u4(0),
            u4(good), u4(good),
        ])
        fh.write(rec(2, 20, wrr))
        good_total += good
        part_total += len(w.die)

    pcr = b"".join([
        u1(1), u1(1), u4(part_total), u4(0), u4(0), u4(good_total), u4(good_total),
    ])
    fh.write(rec(1, 30, pcr))
    fh.write(rec(1, 20, u4(start_ts) + c1(" ") + c("demo complete") + c("")))


def generate(out_dir: Path, seed: int = 20261118) -> list[Path]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("LOT-A-2646", 0.000, 0),
        ("LOT-B-2646", 0.0045, 1),
        ("LOT-C-2646", -0.0055, 2),
    ]
    base = datetime(2026, 11, 2, 8, 0, 0)
    paths = []
    for i, (lot_id, offset, idx) in enumerate(specs):
        lot = make_lot(rng, idx, lot_id, 5, offset, base + timedelta(days=i))
        path = out_dir / f"{lot_id}.stdf"
        with path.open("wb") as fh:
            far = u1(2) + u1(4)
            fh.write(rec(0, 10, far))
            fh.write(rec(0, 20, u4(int(base.timestamp())) + c("python generate_demo_stdf.py")))
            write_lot(fh, lot, idx)
        paths.append(path)
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB, "
              f"{sum(len(w.die) for w in lot.wafers)} die)")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    generate(args.out, args.seed)

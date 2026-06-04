#!/usr/bin/env python3
"""
Parse ECOS catalog files to HyFI input format.

Two conversions are supported:

1. ConsolidatedMergeCat  (pipe-delimited, comment lines start with #)
   -> HyFI hypocenter file (comma-separated)

   Column mapping (source name -> HyFI):
     KP-ID       -> ID
     Pref-lat    -> LAT      [deg]
     Pref-lon    -> LON      [deg]
     Pref-dep    -> DEPTH    [km, positive below sea level]
     CHX         -> X        [m, CH1903+]
     CHY         -> Y        [m, CH1903+]
     -Pref-dep * 1000        -> Z   [m, negative below sea level]
     Err-X-m     -> EX       [m]
     Err-Y-m     -> EY       [m]
     Err-Z-m     -> EZ       [m]
     YYYY        -> YR
     MM          -> MO
     DD          -> DY
     HH          -> HR
     MI          -> MI
     SS          -> SC
     mag         -> MAG
     (no empty columns — all available in source)

2. AssociateFM  (pipe-delimited, comment lines start with #)
   -> HyFI focal mechanism file (semicolon-separated)

   Column mapping (source name -> HyFI):
     EvID_KP     -> ID
     Lat         -> LAT      [deg]
     Lon         -> LON      [deg]
     Dep         -> DEPTH    [km, positive below sea level]
     CH-X        -> X        [m, CH1903+]
     CH-Y        -> Y        [m, CH1903+]
     -Dep * 1000             -> Z   [m, negative below sea level]
     YYYY        -> YR
     MM          -> MO
     DD          -> DY
     HH          -> HR
     MI          -> MI
     SS.S        -> SC
     Mag         -> MAG
     AP          -> A
     S1          -> Strike1
     D1          -> Dip1
     R1          -> Rake1
     S2          -> Strike2
     D2          -> Dip2
     R2          -> Rake2
     Paz         -> Pazim
     Ppl         -> Pdip
     Taz         -> Tazim
     Tpl         -> Tdip
     FMQ         -> Q
     FMT         -> Type
Column names are read from the header line embedded in the comment block
(the last #-prefixed line that contains pipe separators).

Usage:
    python parse_ECOS_to_HyFI.py --hypo <ConsolidatedMergeCat.csv> --focals <AssociateFM.csv>

    # either argument is optional:
    python parse_ECOS_to_HyFI.py --hypo <ConsolidatedMergeCat.csv>
    python parse_ECOS_to_HyFI.py --focals <AssociateFM.csv>
"""

import sys
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Hypocenter: source column name -> HyFI column name
# ---------------------------------------------------------------------------
HYPO_MAP = {
    "KP-ID":   "ID",
    "Pref-lat": "LAT",
    "Pref-lon": "LON",
    "Pref-dep": "DEPTH",
    "CHX":     "X",
    "CHY":     "Y",
    "Err-X-m": "EX",
    "Err-Y-m": "EY",
    "Err-Z-m": "EZ",
    "YYYY":    "YR",
    "MM":      "MO",
    "DD":      "DY",
    "HH":      "HR",
    "MI":      "MI",
    "SS":      "SC",
    "mag":     "MAG",
}

# ---------------------------------------------------------------------------
# Focal mechanism: source column name -> HyFI column name
# ---------------------------------------------------------------------------
FM_MAP = {
    "EvID_KP": "ID",
    "Lat":     "LAT",
    "Lon":     "LON",
    "Dep":     "DEPTH",
    "CH-X":    "X",
    "CH-Y":    "Y",
    "YYYY":    "YR",
    "MM":      "MO",
    "DD":      "DY",
    "HH":      "HR",
    "MI":      "MI",
    "SS.S":    "SC",
    "Mag":     "MAG",
    "AP":      "A",
    "S1":      "Strike1",
    "D1":      "Dip1",
    "R1":      "Rake1",
    "S2":      "Strike2",
    "D2":      "Dip2",
    "R2":      "Rake2",
    "Paz":     "Pazim",
    "Ppl":     "Pdip",
    "Taz":     "Tazim",
    "Tpl":     "Tdip",
    "FMQ":     "Q",
    "FMT":     "Type",
}

# Integer columns in HyFI output
INT_COLS = {"YR", "MO", "DY", "HR", "MI", "A", "Q"}


def _extract_column_names(path: Path) -> list[str]:
    """
    Return the column names embedded in the comment block.
    The header line is the last #-prefixed line that contains '|'.
    """
    header_line = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") and "|" in line:
                header_line = line
            elif not line.startswith("#"):
                break
    if header_line is None:
        raise ValueError(f"No pipe-separated header line found in comments of {path}")
    # Strip leading '#' and split on '|', trimming whitespace from each name
    return [c.strip() for c in header_line.lstrip("#").split("|")]


def _read_src(path: Path) -> pd.DataFrame:
    """Read pipe-delimited file, skipping comment lines, using embedded column names."""
    col_names = _extract_column_names(path)
    df = pd.read_csv(
        path, sep="|", comment="#", header=None,
        names=col_names, dtype=str, skipinitialspace=True,
    )
    # Drop any trailing empty column produced by a trailing '|'
    df = df.loc[:, df.columns.notna() & (df.columns != "")]
    return df


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.str.strip(), errors="coerce")


def parse_hypo(input_file: str, output_file: str = None) -> str:
    """Convert ConsolidatedMergeCat to HyFI hypocenter CSV."""
    inp = Path(input_file)
    out = Path(output_file) if output_file else inp.with_name(inp.stem + "_HyFI.csv")

    print(f"[hypo] Reading: {inp}")
    src = _read_src(inp)
    print(f"       {len(src)} events, {src.shape[1]} columns")
    print(f"       Columns: {list(src.columns)}")

    df = pd.DataFrame()
    for src_col, hyfi_col in HYPO_MAP.items():
        if src_col not in src.columns:
            raise KeyError(f"Expected column '{src_col}' not found. Available: {list(src.columns)}")
        series = _num(src[src_col]) if hyfi_col != "ID" else src[src_col].str.strip()
        if hyfi_col in INT_COLS:
            series = series.astype("Int64")
        df[hyfi_col] = series

    df["Z"] = -df["DEPTH"] * 1000.0
    # Reorder to canonical HyFI column order
    ordered = ["ID", "LAT", "LON", "DEPTH", "X", "Y", "Z",
               "EX", "EY", "EZ", "YR", "MO", "DY", "HR", "MI", "SC", "MAG"]

    df = df[ordered]

    df.to_csv(out, index=False)
    print(f"       Written: {out}")
    return str(out)


def parse_fm(input_file: str, output_file: str = None) -> str:
    """Convert AssociateFM to HyFI focal mechanism CSV (semicolon-separated)."""
    inp = Path(input_file)
    out = Path(output_file) if output_file else inp.with_name(inp.stem + "_HyFI.csv")

    print(f"[fm]   Reading: {inp}")
    src = _read_src(inp)
    print(f"       {len(src)} events, {src.shape[1]} columns")
    print(f"       Columns: {list(src.columns)}")

    df = pd.DataFrame()
    for src_col, hyfi_col in FM_MAP.items():
        if src_col not in src.columns:
            raise KeyError(f"Expected column '{src_col}' not found. Available: {list(src.columns)}")
        series = _num(src[src_col]) if hyfi_col != "ID" else src[src_col].str.strip()
        if hyfi_col in INT_COLS:
            series = series.astype("Int64")
        elif hyfi_col == "Type":
            series = src[src_col].str.strip()
        df[hyfi_col] = series

    df["Z"] = -df["DEPTH"] * 1000.0
    # Reorder to canonical HyFI FM column order
    ordered = ["ID", "LAT", "LON", "DEPTH", "X", "Y", "Z",
               "YR", "MO", "DY", "HR", "MI", "SC", "MAG",
               "A", "Strike1", "Dip1", "Rake1", "Strike2", "Dip2", "Rake2",
               "Pazim", "Pdip", "Tazim", "Tdip", "Q", "Type"]
    df = df[ordered]

    df.to_csv(out, index=False)
    print(f"       Written: {out}")
    return str(out)


def _detect_type(path: Path) -> str:
    """Return 'hypo' or 'fm' based on filename keywords, then column count."""
    name = path.name.lower()
    if "associatefm" in name:
        return "fm"
    if "mergecat" in name:
        return "hypo"
    # Fallback: count columns in the embedded header line
    try:
        cols = _extract_column_names(path)
        return "hypo" if len(cols) > 40 else "fm"
    except ValueError:
        return "hypo"


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Parse ECOS catalog files to HyFI input format."
    )
    parser.add_argument("--hypo",   metavar="FILE", help="ConsolidatedMergeCat CSV -> HyFI hypocenter CSV")
    parser.add_argument("--focals", metavar="FILE", help="AssociateFM CSV -> HyFI focal mechanism CSV")
    args = parser.parse_args()

    if not args.hypo and not args.focals:
        parser.print_help()
        sys.exit(1)

    if args.hypo:
        parse_hypo(args.hypo)
    if args.focals:
        parse_fm(args.focals)


if __name__ == "__main__":
    main()

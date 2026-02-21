"""
Merge per-width convergence and LSI summaries for cross-width plotting.
Output: convergence_by_width.csv, lsi_proxy_by_width.csv (with width column).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BASE = Path("experiments/summaries")


def main() -> None:
    widths = [(1.0, "convergence.csv", "lsi_proxy.csv"), (0.1, "convergence_w0.1.csv", "lsi_proxy_w0.1.csv"), (0.01, "convergence_w0.01.csv", "lsi_proxy_w0.01.csv")]
    conv_dfs = []
    lsi_dfs = []
    for w, conv_name, lsi_name in widths:
        cp = BASE / conv_name
        lp = BASE / lsi_name
        if cp.exists():
            df = pd.read_csv(cp)
            df["width"] = w
            conv_dfs.append(df)
        if lp.exists():
            df = pd.read_csv(lp)
            df["width"] = w
            lsi_dfs.append(df)
    if conv_dfs:
        merged_conv = pd.concat(conv_dfs, ignore_index=True)
        out_conv = BASE / "convergence_by_width.csv"
        merged_conv.to_csv(out_conv, index=False)
        print("Wrote", out_conv)
    if lsi_dfs:
        merged_lsi = pd.concat(lsi_dfs, ignore_index=True)
        out_lsi = BASE / "lsi_proxy_by_width.csv"
        merged_lsi.to_csv(out_lsi, index=False)
        print("Wrote", out_lsi)


if __name__ == "__main__":
    main()

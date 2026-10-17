"""Generate synthetic but realistic freight rate history."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "sample_rates.csv"

LANES = [
    ("CNSHA", "USLAX", "ocean", 1800),
    ("CNSHA", "NLRTM", "ocean", 1600),
    ("CNNGB", "USNYC", "ocean", 2100),
    ("DEHAM", "USNYC", "ocean", 1400),
    ("CNSHA", "USLAX", "air", 4.8),  # USD per kg proxy scaled later
    ("HKHKG", "LHR", "air", 5.2),
]


def main() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", "2025-06-30", freq="D")
    rows = []
    for origin, dest, mode, base in LANES:
        for d in dates:
            seasonal = 120 * np.sin(2 * np.pi * d.dayofyear / 365)
            trend = 0.15 * (d - dates[0]).days
            noise = float(rng.normal(0, 60 if mode == "ocean" else 0.25))
            if mode == "air":
                rate = (base * 1000) + seasonal * 0.5 + trend * 0.05 + noise * 40
            else:
                rate = base + seasonal + trend + noise
            rows.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "origin": origin,
                    "dest": dest,
                    "mode": mode,
                    "rate": round(float(rate), 2),
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()

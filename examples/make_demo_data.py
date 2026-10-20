"""Create tiny, artificial Parquet inputs to check NextGen integration and training.

These samples are not a real forecast benchmark. They match forecast_backbones/m4_loop/config/smoke.yaml,
not the production default 672/192-point windows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_frame(start: int, count: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for index in range(start, start + count):
        capacity = float(40 + 10 * (index % 3))
        irradiance = rng.uniform(0, 1000, 24).astype(np.float32)
        temperature = rng.normal(25, 4, 24).astype(np.float32)
        power = capacity * irradiance / 1000 * (1 - 0.004 * (temperature - 25))
        power = np.clip(power + rng.normal(0, capacity * 0.01, 24), 0, capacity)
        prior = np.clip(power[16:] + rng.normal(0, capacity * 0.08, 8), 0, capacity)
        rows.append(
            {
                "general_ghi": irradiance[:16].tolist(),
                "general_temp": temperature[:16].tolist(),
                "general_ghi_predict": irradiance[16:].tolist(),
                "general_temp_predict": temperature[16:].tolist(),
                "observe_power": power[:16].tolist(),
                "chronos": prior.tolist(),
                "observe_power_future": power[16:].tolist(),
                "cap_power_on": capacity,
                "station": f"synthetic_{index % 3}",
                "timestamp_win": pd.Timestamp("2025-01-01") + pd.Timedelta(days=index),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2025)
    start = 0
    for name, count in [("train", 24), ("valid", 8), ("test", 8)]:
        destination = args.out / f"{name}.parquet"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        make_frame(start, count, rng).to_parquet(destination, index=False)
        print(f"{name}: {count} synthetic rows -> {destination}")
        start += count


if __name__ == "__main__":
    main()

"""Download the Brazilian E-Commerce Public Dataset by Olist from Kaggle.

Public dataset, no API token required. Uses kagglehub, which caches the
download under ~/.cache/kagglehub and copies the CSVs into data/raw/.
"""

import shutil
from pathlib import Path

import kagglehub

DATASET = "olistbr/brazilian-ecommerce"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = Path(kagglehub.dataset_download(DATASET))

    csv_files = sorted(cache_path.glob("*.csv"))
    for csv_file in csv_files:
        shutil.copy2(csv_file, RAW_DIR / csv_file.name)

    print(f"Copied {len(csv_files)} files to {RAW_DIR}: {[f.name for f in csv_files]}")


if __name__ == "__main__":
    main()

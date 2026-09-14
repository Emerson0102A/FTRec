from __future__ import annotations

import csv
import gzip
from pathlib import Path


DOMAIN_FILES = {
    "Health": "Health_and_Household.csv.gz",
    "Clothing": "Clothing_Shoes_and_Jewelry.csv.gz",
    "Beauty": "Beauty_and_Personal_Care.csv.gz",
    "Grocery": "Grocery_and_Gourmet_Food.csv.gz",
    "Sports": "Sports_and_Outdoors.csv.gz",
}


def write_gzip_csv(path: Path, rows: list[tuple[str, str, str, str]], *, header=None) -> None:
    fields = header or ["user_id", "parent_asin", "rating", "timestamp"]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def write_amazon5_fixture(root: Path, rows_by_domain=None, *, header_by_domain=None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows_by_domain = rows_by_domain or {}
    header_by_domain = header_by_domain or {}
    for index, (domain, filename) in enumerate(DOMAIN_FILES.items()):
        rows = rows_by_domain.get(
            domain,
            [(f"base-{domain}", f"item-{domain}", "5.0", str(1000 + index))],
        )
        write_gzip_csv(root / filename, rows, header=header_by_domain.get(domain))
    return root


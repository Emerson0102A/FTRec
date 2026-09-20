"""Build an Amazon metadata catalog aligned to released MDSR item IDs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping


class RestrictedMappingUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in {
            ("numpy", "dtype"),
            ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "scalar"),
        }:
            import numpy as np

            return np.dtype if name == "dtype" else np.core.multiarray.scalar
        raise pickle.UnpicklingError(f"blocked pickle global: {module}.{name}")


def load_mdsr_mappings(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        mappings = RestrictedMappingUnpickler(handle).load()
    required = {"domain", "domain_offset", "item"}
    if not isinstance(mappings, dict) or not required.issubset(mappings):
        raise ValueError(f"invalid MDSR mapping; expected keys {sorted(required)}")
    ids = sorted(int(value) for value in mappings["item"].values())
    if ids != list(range(len(ids))):
        raise ValueError("MDSR item IDs must be unique and contiguous from zero")
    return mappings


def _text(value: Any) -> str:
    return "" if value is None else " ".join(str(value).replace("\x00", " ").split())


def _list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [cleaned for item in value if (cleaned := _text(item))]


@dataclass(frozen=True)
class CatalogBuildResult:
    output_path: Path
    manifest_path: Path
    item_count: int
    metadata_found: int
    metadata_missing: int
    catalog_sha256: str


def _gzip_writer(raw: BinaryIO) -> io.TextIOWrapper:
    return io.TextIOWrapper(
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0),
        encoding="utf-8",
        newline="\n",
    )


def iter_catalog_rows(
    dataset_dir: str | Path, mappings: Mapping[str, Any]
) -> Iterator[dict[str, Any]]:
    dataset = Path(dataset_dir)
    domains = {str(name): int(value) for name, value in mappings["domain"].items()}
    offsets = {int(key): tuple(map(int, value)) for key, value in mappings["domain_offset"].items()}
    asin_to_id = {str(key): int(value) for key, value in mappings["item"].items()}
    id_to_asin = [""] * len(asin_to_id)
    for asin, zero_id in asin_to_id.items():
        id_to_asin[zero_id] = asin
    found = bytearray(len(id_to_asin))
    for domain, domain_id in sorted(domains.items(), key=lambda pair: pair[1]):
        path = dataset / f"meta_{domain}.jsonl.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        start, end = offsets[domain_id]
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                asin = _text(raw.get("parent_asin"))
                zero_id = asin_to_id.get(asin)
                if zero_id is None or found[zero_id] or not start <= zero_id < end:
                    continue
                found[zero_id] = 1
                details = raw.get("details") if isinstance(raw.get("details"), dict) else {}
                yield {
                    "item_id": zero_id + 1,
                    "domain_id": domain_id,
                    "domain": domain,
                    "parent_asin": asin,
                    "title": _text(raw.get("title")),
                    "store": _text(raw.get("store")),
                    "main_category": _text(raw.get("main_category")),
                    "categories": _list(raw.get("categories")),
                    "features": _list(raw.get("features")),
                    "description": _list(raw.get("description")),
                    "details": {
                        key: val for raw_key, raw_val in details.items()
                        if (key := _text(raw_key)) and (val := _text(raw_val))
                    },
                    "metadata_found": True,
                }
    ranges = sorted(
        (offsets[domain_id][0], offsets[domain_id][1], domain, domain_id)
        for domain, domain_id in domains.items()
    )
    for zero_id, was_found in enumerate(found):
        if was_found:
            continue
        domain, domain_id = next(
            (name, value) for start, end, name, value in ranges if start <= zero_id < end
        )
        yield {
            "item_id": zero_id + 1, "domain_id": domain_id, "domain": domain,
            "parent_asin": id_to_asin[zero_id], "title": "", "store": "",
            "main_category": "", "categories": [], "features": [], "description": [],
            "details": {}, "metadata_found": False,
        }


def build_catalog(
    dataset_dir: str | Path, mappings_path: str | Path, output_path: str | Path
) -> CatalogBuildResult:
    output = Path(output_path)
    if output.suffixes[-2:] != [".jsonl", ".gz"]:
        raise ValueError("catalog output must end in .jsonl.gz")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix("").with_suffix(".manifest.json")
    temporary = output.with_name(output.name + ".tmp")
    mappings = load_mdsr_mappings(mappings_path)
    digest = hashlib.sha256()
    rows = found = 0
    with temporary.open("wb") as raw:
        with _gzip_writer(raw) as stream:
            for row in iter_catalog_rows(dataset_dir, mappings):
                line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                digest.update(line.encode("utf-8"))
                stream.write(line)
                rows += 1
                found += int(row["metadata_found"])
    temporary.replace(output)
    item_count = len(mappings["item"])
    if rows != item_count:
        raise RuntimeError(f"catalog has {rows} rows; mapping has {item_count} items")
    manifest = {
        "format": "ftrec-item-catalog", "version": 1,
        "item_id_convention": "one-based; zero is padding", "item_count": item_count,
        "metadata_found": found, "metadata_missing": item_count - found,
        "domain_to_id": {str(key): int(value) for key, value in mappings["domain"].items()},
        "catalog_sha256": digest.hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return CatalogBuildResult(output, manifest_path, item_count, found, item_count - found, digest.hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--mappings", type=Path, default=Path("data/MDSR-Amazon/mappings.pkl"))
    parser.add_argument("--output", type=Path, default=Path("data/attribute_experiment/catalog.jsonl.gz"))
    args = parser.parse_args()
    result = build_catalog(args.dataset_dir, args.mappings, args.output)
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in asdict(result).items()}, indent=2))


if __name__ == "__main__":
    main()

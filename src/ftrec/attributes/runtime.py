"""Preflight GPU, dependencies, hub mirror, and LLM2Attr checkpoints."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .checkpoints import resolve_checkpoint_paths
from .llm2attr import DEFAULT_SERVER_ROOT


REQUIRED = (
    ("transformers", "transformers"), ("peft", "peft"),
    ("llm2vec", "llm2vec"), ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
)

PINNED_RUNTIME = {
    "transformers": "4.44.2",
    "peft": "0.18.1",
    "llm2vec": "0.2.3",
}


def _packages() -> tuple[dict[str, str], list[str]]:
    installed: dict[str, str] = {}
    missing: list[str] = []
    for distribution, module in REQUIRED:
        if importlib.util.find_spec(module) is None:
            missing.append(distribution)
            continue
        try:
            installed[distribution] = version(distribution)
        except PackageNotFoundError:
            installed[distribution] = "unknown"
    return installed, missing


def recommended_batch_size(memory_gib: float) -> int:
    if memory_gib >= 70:
        return 32
    if memory_gib >= 40:
        return 24
    if memory_gib >= 20:
        return 16
    return 8


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    result: dict[str, Any] = {
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        "model": args.model,
    }
    try:
        paths = resolve_checkpoint_paths(args.llm2attr_root)
        result["checkpoints"] = {
            "root": str(paths.root), "mntp": str(paths.mntp),
            "attribute": str(paths.attribute),
        }
        configs = {
            label: json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
            for label, path in (("mntp", paths.mntp), ("attribute", paths.attribute))
        }
        result["checkpoint_peft_versions"] = {
            label: config.get("peft_version") for label, config in configs.items()
        }
        for label, config in configs.items():
            if config.get("base_model_name_or_path") != args.model:
                errors.append(f"{label} checkpoint expects {config.get('base_model_name_or_path')!r}")
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))

    installed, missing = _packages()
    result["packages"] = installed
    if missing:
        errors.append("missing Python packages: " + ", ".join(missing))
    for package, expected_version in PINNED_RUNTIME.items():
        actual_version = installed.get(package)
        if actual_version is not None and actual_version != expected_version:
            errors.append(
                f"{package} must be {expected_version} for this LLM2Attr "
                f"runtime; installed version is {actual_version}"
            )
    if not missing:
        for distribution, module in REQUIRED:
            try:
                importlib.import_module(module)
            except Exception as exc:
                errors.append(
                    f"cannot import {distribution} {installed.get(distribution)}: "
                    f"{type(exc).__name__}: {exc}"
                )
    try:
        import torch

        result["torch"] = {
            "version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
        if not torch.cuda.is_available():
            errors.append("CUDA is not visible to PyTorch")
        else:
            properties = torch.cuda.get_device_properties(0)
            memory_gib = properties.total_memory / 1024**3
            result["gpu"] = {
                "name": properties.name, "memory_gib": round(memory_gib, 2),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                "recommended_batch_size": recommended_batch_size(memory_gib),
                "recommended_dtype": "bfloat16", "recommended_attention": "sdpa",
            }
            if not torch.cuda.is_bf16_supported():
                errors.append("GPU/PyTorch does not support BF16")
            if "4090" in properties.name and memory_gib > 30:
                warnings.append(
                    "GPU name contains 4090 but visible memory exceeds the usual card size; "
                    "confirm the device with nvidia-smi"
                )
    except ImportError:
        errors.append("PyTorch is not installed")

    if not missing and not args.skip_hub_check:
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(args.model)
            result["hub_model_check"] = {"status": "ok", "model_type": config.model_type}
        except Exception as exc:
            errors.append(f"cannot resolve base model through configured hub: {exc}")
    elif args.skip_hub_check:
        result["hub_model_check"] = {"status": "skipped"}
    expected = result.get("checkpoint_peft_versions", {}).get("attribute")
    if expected and installed.get("peft") and expected != installed["peft"]:
        warnings.append(f"checkpoint PEFT is {expected}; installed PEFT is {installed['peft']}")
    result.update(warnings=warnings, errors=errors, status="pass" if not errors else "fail")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm2attr-root", type=Path, default=DEFAULT_SERVER_ROOT)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--skip-hub-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run_preflight(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
